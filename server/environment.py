"""CrisprEnv v2 — Tool-based CRISPR gene editing environment."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .graders import grade_multi_repair, grade_precision_editing, grade_single_target
from .models import (
    CorrectionResult,
    EnvironmentState,
    GuideDesign,
    OffTargetHit,
    ToolResult,
)
from .reward import TOOL_COSTS, compute_step_reward
from .simulation import (
    analyze_sequence_region,
    design_guide_at_pam,
    evaluate_guide,
    scan_off_targets,
    search_pam_sites,
    simulate_edit,
)
from .tasks import TASK_REGISTRY, TaskScenario


class CrisprEnv:
    """Tool-based CRISPR gene editing environment.

    The agent uses bioinformatics tools to investigate sequences, design guides,
    evaluate safety, and apply edits under resource constraints.
    """

    def __init__(self, task_level: str = "single_target", seed: int = 42):
        if task_level not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_level}. Available: {list(TASK_REGISTRY)}")
        self.task_level = task_level
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._scenario: Optional[TaskScenario] = None
        self._steps_taken = 0
        self._done = False
        self._budget_used = 0
        self._edits_applied = 0
        self._corrections: List[CorrectionResult] = []
        self._off_target_damage: List[OffTargetHit] = []
        self._tool_history: List[ToolResult] = []
        self._last_tool: Optional[str] = None
        self._last_tool_output: Optional[str] = None
        self._last_tool_error: Optional[str] = None
        self._designed_guides: Dict[str, GuideDesign] = {}
        self._guide_counter = 0
        self._explored_regions: set = set()
        self._searched_pams: set = set()
        self._final_score: Optional[float] = None

    def reset(self) -> EnvironmentState:
        generator = TASK_REGISTRY[self.task_level]["generator"]
        self._rng = np.random.default_rng(self._seed)
        self._scenario = generator(self._rng)
        self._steps_taken = 0
        self._done = False
        self._budget_used = 0
        self._edits_applied = 0
        self._corrections = []
        self._off_target_damage = []
        self._tool_history = []
        self._last_tool = None
        self._last_tool_output = None
        self._last_tool_error = None
        self._designed_guides = {}
        self._guide_counter = 0
        self._explored_regions = set()
        self._searched_pams = set()
        self._final_score = None
        return self.state()

    def state(self) -> EnvironmentState:
        if self._scenario is None:
            raise RuntimeError("Call reset() first.")
        s = self._scenario
        return EnvironmentState(
            task_description=s.description,
            task_type=s.task_type,
            target_gene_id=s.gene_id,
            target_gene_length=len(s.gene_sequence),
            known_mutations=s.mutations,
            regulatory_regions=s.regulatory_regions,
            experiment_budget=s.experiment_budget - self._budget_used,
            max_steps=s.max_steps,
            steps_taken=self._steps_taken,
            edits_applied=self._edits_applied,
            last_tool=self._last_tool,
            last_tool_output=self._last_tool_output,
            last_tool_error=self._last_tool_error,
            tool_history=self._tool_history[-20:],  # keep last 20 for context
            corrections_made=self._corrections,
            off_target_damage=self._off_target_damage,
        )

    def step(self, action: str) -> Tuple[EnvironmentState, float, bool, Dict]:
        if self._scenario is None:
            raise RuntimeError("Call reset() first.")
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        self._steps_taken += 1
        self._last_tool_error = None

        # Parse action
        parts = action.strip().split(None, 1)
        tool_name = parts[0] if parts else ""
        args_str = parts[1] if len(parts) > 1 else ""

        info: Dict = {"action": tool_name}

        if tool_name not in TOOL_COSTS:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = f"Unknown tool: '{tool_name}'. Available: {list(TOOL_COSTS.keys())}"
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, self._last_tool_error)
            return self._check_done(reward, info)

        # Check budget
        cost = TOOL_COSTS[tool_name]
        remaining = self._scenario.experiment_budget - self._budget_used
        if cost > remaining:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = f"Not enough budget. {tool_name} costs {cost} credits, you have {remaining} remaining."
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, self._last_tool_error)
            return self._check_done(reward, info)

        # Dispatch
        try:
            output, new_info = self._dispatch(tool_name, args_str)
            self._budget_used += cost
            self._last_tool = tool_name
            self._last_tool_output = output
            explored_new = new_info
            reward = compute_step_reward(tool_name, True, explored_new)
            info.update({"credits_spent": cost, "budget_remaining": remaining - cost})
            self._record_history(tool_name, args_str, output[:200] if output else "")
        except Exception as e:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = str(e)
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, f"ERROR: {e}")

        return self._check_done(reward, info)

    def close(self) -> None:
        self._scenario = None
        self._done = True

    # ── Tool dispatch ────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, args_str: str) -> Tuple[str, bool]:
        """Dispatch tool call. Returns (output_text, explored_new_info)."""
        s = self._scenario

        if tool_name == "analyze_sequence":
            return self._handle_analyze(args_str)

        elif tool_name == "search_pam_sites":
            return self._handle_search_pam(args_str)

        elif tool_name == "design_guide":
            return self._handle_design_guide(args_str)

        elif tool_name == "evaluate_guide":
            return self._handle_evaluate_guide(args_str)

        elif tool_name == "off_target_scan":
            return self._handle_off_target_scan(args_str)

        elif tool_name == "apply_edit":
            return self._handle_apply_edit(args_str)

        elif tool_name == "check_edit_result":
            return self._handle_check_result()

        elif tool_name == "submit_solution":
            return self._handle_submit()

        raise ValueError(f"Unhandled tool: {tool_name}")

    def _handle_analyze(self, args_str: str) -> Tuple[str, bool]:
        parts = args_str.split()
        if len(parts) != 2:
            raise ValueError("Usage: analyze_sequence <start> <end>")
        start, end = int(parts[0]), int(parts[1])
        gene = self._scenario.gene_sequence
        if start < 0 or end > len(gene) or start >= end:
            raise ValueError(f"Invalid range. Gene length is {len(gene)}.")

        key = (start // 50, end // 50)
        new_info = key not in self._explored_regions
        self._explored_regions.add(key)

        result = analyze_sequence_region(gene, start, end)
        lines = [
            f"Region: {result['region']} ({result['length']}bp)",
            f"Sequence: {result['sequence']}",
            f"GC content: {result['gc_content']}",
            f"Max homopolymer run: {result['max_homopolymer_run']}",
            f"Dinucleotide repeats: {result['dinucleotide_repeats']}",
            f"Self-complementarity: {result['self_complementarity']}",
            f"Structure risk: {result['structure_risk']}",
        ]
        return "\n".join(lines), new_info

    def _handle_search_pam(self, args_str: str) -> Tuple[str, bool]:
        pattern = args_str.strip().upper()
        if not pattern:
            raise ValueError("Usage: search_pam_sites <pattern> (e.g., NGG)")
        if not all(c in "ACGTNR" for c in pattern):
            raise ValueError(f"Invalid PAM pattern: {pattern}. Use A/C/G/T/N/R.")

        new_info = pattern not in self._searched_pams
        self._searched_pams.add(pattern)

        sites = search_pam_sites(self._scenario.gene_sequence, pattern)
        if not sites:
            return f"No PAM sites found for pattern '{pattern}'.", new_info

        # Show sites near mutations first, then others
        mut_positions = [m.position for m in self._scenario.mutations]

        def sort_key(s):
            return min(abs(s.position - mp) for mp in mut_positions)

        sites.sort(key=sort_key)
        lines = [f"Found {len(sites)} PAM sites for pattern '{pattern}':"]
        for s in sites[:15]:  # show top 15
            dist = min(abs(s.position - mp) for mp in mut_positions)
            lines.append(
                f"  pos={s.position} strand={s.strand} seq={s.sequence} "
                f"dist_to_nearest_mutation={dist}"
            )
        if len(sites) > 15:
            lines.append(f"  ... and {len(sites) - 15} more")
        return "\n".join(lines), new_info

    def _handle_design_guide(self, args_str: str) -> Tuple[str, bool]:
        parts = args_str.split()
        if len(parts) != 2:
            raise ValueError("Usage: design_guide <pam_position> <strand>")
        pam_pos = int(parts[0])
        strand = parts[1]
        if strand not in ("+", "-"):
            raise ValueError("Strand must be '+' or '-'.")

        self._guide_counter += 1
        guide = design_guide_at_pam(
            self._scenario.gene_sequence, pam_pos, strand,
            self._scenario.mutations, self._guide_counter,
        )
        if guide is None:
            raise ValueError(f"Cannot design guide at position {pam_pos} strand {strand} — out of bounds.")

        self._designed_guides[guide.sequence] = guide

        lines = [
            f"Guide designed: {guide.guide_id}",
            f"  Sequence: {guide.sequence}",
            f"  PAM position: {guide.pam_position}, strand: {guide.strand}",
            f"  Distance to nearest mutation: {guide.distance_to_nearest_mutation}bp",
            f"  GC content: {guide.gc_content}",
            f"  On-target score: {guide.on_target_score}",
        ]
        return "\n".join(lines), True

    def _handle_evaluate_guide(self, args_str: str) -> Tuple[str, bool]:
        seq = args_str.strip().upper()
        if not seq or not all(c in "ACGT" for c in seq):
            raise ValueError("Usage: evaluate_guide <guide_sequence> (20nt A/C/G/T)")
        if len(seq) < 15 or len(seq) > 25:
            raise ValueError("Guide sequence should be 15-25 nucleotides.")

        ev = evaluate_guide(seq)
        lines = [
            f"Guide evaluation for: {seq}",
            f"  GC content: {ev.gc_content} ({ev.gc_quality})",
            f"  Max homopolymer run: {ev.homopolymer_max_run}",
            f"  Self-complementarity: {ev.self_complementarity}",
            f"  Overall quality: {ev.overall_quality}",
        ]
        return "\n".join(lines), True

    def _handle_off_target_scan(self, args_str: str) -> Tuple[str, bool]:
        seq = args_str.strip().upper()
        if not seq or not all(c in "ACGT" for c in seq):
            raise ValueError("Usage: off_target_scan <guide_sequence>")

        s = self._scenario
        hits = scan_off_targets(
            seq, s.genome_context, s.gene_offset, len(s.gene_sequence),
            s.regulatory_regions,
        )
        if not hits:
            return "No off-target sites found (<=3 mismatches). Guide appears safe.", True

        lines = [f"Found {len(hits)} potential off-target site(s):"]
        for h in hits:
            reg_flag = " *** IN REGULATORY REGION ***" if h.in_regulatory_region else ""
            lines.append(
                f"  pos={h.position} mismatches={h.mismatches} "
                f"risk={h.risk_level}{reg_flag}"
            )
        return "\n".join(lines), True

    def _handle_apply_edit(self, args_str: str) -> Tuple[str, bool]:
        parts = args_str.split()
        if len(parts) != 2:
            raise ValueError("Usage: apply_edit <guide_sequence> <target_position>")
        seq = parts[0].upper()
        target_pos = int(parts[1])

        if not all(c in "ACGT" for c in seq):
            raise ValueError("Guide sequence must contain only A/C/G/T.")

        s = self._scenario

        # Get on-target score if we have it from a designed guide
        on_target = 0.5  # default
        if seq in self._designed_guides:
            on_target = self._designed_guides[seq].on_target_score

        # Get known off-target hits
        known_hits = scan_off_targets(
            seq, s.genome_context, s.gene_offset, len(s.gene_sequence),
            s.regulatory_regions,
        )

        corrections, damage = simulate_edit(
            seq, target_pos, s.mutations, known_hits, on_target, self._rng,
        )

        self._edits_applied += 1
        self._corrections.extend(corrections)
        self._off_target_damage.extend(damage)

        lines = ["Edit applied."]
        for c in corrections:
            status = "CORRECTED" if c.corrected else "FAILED"
            lines.append(f"  Mutation at {c.mutation_position}: {status}")
        if damage:
            lines.append(f"  Off-target damage: {len(damage)} site(s) affected")
            for d in damage:
                reg = " (REGULATORY REGION!)" if d.in_regulatory_region else ""
                lines.append(f"    pos={d.position} mismatches={d.mismatches}{reg}")
        else:
            lines.append("  No off-target damage detected.")

        # Auto-end if all mutations corrected
        all_corrected = all(
            any(c.corrected and c.mutation_position == m.position for c in self._corrections)
            for m in s.mutations
        )
        if all_corrected:
            lines.append("All mutations corrected!")

        return "\n".join(lines), True

    def _handle_check_result(self) -> Tuple[str, bool]:
        s = self._scenario
        lines = ["Current edit results:"]
        for m in s.mutations:
            corrected = any(
                c.corrected and c.mutation_position == m.position
                for c in self._corrections
            )
            lines.append(f"  Mutation at {m.position} ({m.ref_base}->{m.alt_base}): {'CORRECTED' if corrected else 'NOT corrected'}")
        lines.append(f"Off-target damage: {len(self._off_target_damage)} site(s)")
        lines.append(f"Budget remaining: {s.experiment_budget - self._budget_used}")
        return "\n".join(lines), False

    def _handle_submit(self) -> Tuple[str, bool]:
        self._done = True
        score = self._compute_final_score()
        self._final_score = score
        return f"Solution submitted. Final score: {score:.4f}", False

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_final_score(self) -> float:
        s = self._scenario
        if s.task_type == "single_target":
            return grade_single_target(
                self._corrections, s.mutations, self._off_target_damage,
                self._budget_used, s.experiment_budget,
            )
        elif s.task_type == "multi_repair":
            return grade_multi_repair(
                self._corrections, s.mutations, self._off_target_damage,
                self._budget_used, s.experiment_budget, self._edits_applied,
            )
        elif s.task_type == "precision_editing":
            return grade_precision_editing(
                self._corrections, s.mutations, self._off_target_damage,
                s.regulatory_regions, self._budget_used, s.experiment_budget,
            )
        return 0.0

    def _check_done(self, reward: float, info: Dict) -> Tuple[EnvironmentState, float, bool, Dict]:
        s = self._scenario
        # Auto-end conditions
        if self._steps_taken >= s.max_steps and not self._done:
            self._done = True
            self._final_score = self._compute_final_score()
            info["max_steps_reached"] = True

        remaining = s.experiment_budget - self._budget_used
        if remaining <= 0 and not self._done:
            # Can still submit or check (free tools), but flag it
            info["budget_exhausted"] = True

        if self._done and self._final_score is not None:
            info["final_score"] = self._final_score

        return self.state(), reward, self._done, info

    def _record_history(self, tool: str, args: str, output_summary: str) -> None:
        self._tool_history.append(ToolResult(
            tool_name=tool,
            args=args,
            output_summary=output_summary[:200],
            step=self._steps_taken,
        ))
