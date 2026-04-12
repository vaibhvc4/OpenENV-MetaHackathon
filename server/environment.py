from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from openenv.core import Environment

from .graders import grade_multi_repair, grade_precision_editing, grade_single_target
from .models import (
    CorrectionResult,
    CrisprAction,
    CrisprObservation,
    CrisprState,
    GuideDesign,
    OffTargetHit,
    ToolResultRecord,
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


class CrisprEnvironment(Environment[CrisprAction, CrisprObservation, CrisprState]):
    """Tool-based CRISPR gene editing environment.

    The agent uses bioinformatics tools to investigate sequences, design guides,
    evaluate safety, and apply edits under resource constraints.
    """

    def __init__(self, task_level: str = "single_target", seed: int = 42):
        if task_level not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_level}. Available: {list(TASK_REGISTRY)}")
        self._task_level = task_level
        self._default_seed = seed
        self._rng = np.random.default_rng(seed)
        self._scenario: Optional[TaskScenario] = None
        self._steps_taken = 0
        self._done = False
        self._budget_used = 0
        self._edits_applied = 0
        self._corrections: List[CorrectionResult] = []
        self._off_target_damage: List[OffTargetHit] = []
        self._tool_history: List[ToolResultRecord] = []
        self._last_tool: Optional[str] = None
        self._last_tool_output: Optional[str] = None
        self._last_tool_error: Optional[str] = None
        self._designed_guides: Dict[str, GuideDesign] = {}
        self._guide_counter = 0
        self._explored_regions: set = set()
        self._searched_pams: set = set()
        self._final_score: Optional[float] = None
        self._episode_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CrisprObservation:
        task_level = kwargs.get("task_level", self._task_level)
        if task_level not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {task_level}. Available: {list(TASK_REGISTRY)}")
        self._task_level = task_level

        actual_seed = seed if seed is not None else self._default_seed
        self._rng = np.random.default_rng(actual_seed)
        self._episode_id = episode_id

        generator = TASK_REGISTRY[self._task_level]["generator"]
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
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: CrisprAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CrisprObservation:
        if self._scenario is None:
            raise RuntimeError("Call reset() first.")
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        command = action.command if isinstance(action, CrisprAction) else str(action)
        self._steps_taken += 1
        self._last_tool_error = None

        parts = command.strip().split(None, 1)
        tool_name = parts[0] if parts else ""
        args_str = parts[1] if len(parts) > 1 else ""

        if tool_name not in TOOL_COSTS:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = f"Unknown tool: '{tool_name}'. Available: {list(TOOL_COSTS.keys())}"
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, self._last_tool_error)
            return self._finalize_step(reward)

        cost = TOOL_COSTS[tool_name]
        remaining = self._scenario.experiment_budget - self._budget_used
        if cost > remaining:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = f"Not enough budget. {tool_name} costs {cost}, you have {remaining}."
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, self._last_tool_error)
            return self._finalize_step(reward)

        try:
            output, explored_new = self._dispatch(tool_name, args_str)
            self._budget_used += cost
            self._last_tool = tool_name
            self._last_tool_output = output
            reward = compute_step_reward(tool_name, True, explored_new)
            self._record_history(tool_name, args_str, output[:200] if output else "")
        except Exception as e:
            self._last_tool = tool_name
            self._last_tool_output = None
            self._last_tool_error = str(e)
            reward = compute_step_reward(tool_name, False, False)
            self._record_history(tool_name, args_str, f"ERROR: {e}")

        return self._finalize_step(reward)

    @property
    def state(self) -> CrisprState:
        return CrisprState(
            episode_id=self._episode_id,
            step_count=self._steps_taken,
            task_type=self._task_level,
            target_gene_id=self._scenario.gene_id if self._scenario else "",
            budget_remaining=(self._scenario.experiment_budget - self._budget_used) if self._scenario else 0,
            edits_applied=self._edits_applied,
            corrections_count=sum(1 for c in self._corrections if c.corrected),
            damage_count=len(self._off_target_damage),
        )

    def close(self) -> None:
        self._scenario = None
        self._done = True

    # ── Observation builders ─────────────────────────────────────────────

    def _build_observation(self, reward: float, done: bool) -> CrisprObservation:
        s = self._scenario
        return CrisprObservation(
            done=done,
            reward=reward,
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
            tool_history=self._tool_history[-20:],
            corrections_made=self._corrections,
            off_target_damage=self._off_target_damage,
            metadata={"final_score": self._final_score} if self._final_score is not None else {},
        )

    def _finalize_step(self, reward: float) -> CrisprObservation:
        s = self._scenario
        if self._steps_taken >= s.max_steps and not self._done:
            self._done = True
            self._final_score = self._compute_final_score()
        return self._build_observation(reward=reward, done=self._done)

    # ── Tool dispatch ────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, args_str: str) -> Tuple[str, bool]:
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
        mut_positions = [m.position for m in self._scenario.mutations]
        sites.sort(key=lambda s: min(abs(s.position - mp) for mp in mut_positions))
        lines = [f"Found {len(sites)} PAM sites for pattern '{pattern}':"]
        for s in sites[:15]:
            dist = min(abs(s.position - mp) for mp in mut_positions)
            lines.append(f"  pos={s.position} strand={s.strand} seq={s.sequence} dist_to_nearest_mutation={dist}")
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
            raise ValueError("Usage: evaluate_guide <guide_sequence> (A/C/G/T)")
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
        hits = scan_off_targets(seq, s.genome_context, s.gene_offset, len(s.gene_sequence), s.regulatory_regions)
        if not hits:
            return "No off-target sites found (<=3 mismatches). Guide appears safe.", True
        lines = [f"Found {len(hits)} potential off-target site(s):"]
        for h in hits:
            reg_flag = " *** IN REGULATORY REGION ***" if h.in_regulatory_region else ""
            lines.append(f"  pos={h.position} mismatches={h.mismatches} risk={h.risk_level}{reg_flag}")
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
        on_target = self._designed_guides[seq].on_target_score if seq in self._designed_guides else 0.5
        known_hits = scan_off_targets(seq, s.genome_context, s.gene_offset, len(s.gene_sequence), s.regulatory_regions)
        corrections, damage = simulate_edit(seq, target_pos, s.mutations, known_hits, on_target, self._rng)
        self._edits_applied += 1
        self._corrections.extend(corrections)
        self._off_target_damage.extend(damage)
        lines = ["Edit applied."]
        for c in corrections:
            lines.append(f"  Mutation at {c.mutation_position}: {'CORRECTED' if c.corrected else 'FAILED'}")
        if damage:
            lines.append(f"  Off-target damage: {len(damage)} site(s) affected")
            for d in damage:
                reg = " (REGULATORY REGION!)" if d.in_regulatory_region else ""
                lines.append(f"    pos={d.position} mismatches={d.mismatches}{reg}")
        else:
            lines.append("  No off-target damage detected.")
        return "\n".join(lines), True

    def _handle_check_result(self) -> Tuple[str, bool]:
        s = self._scenario
        lines = ["Current edit results:"]
        for m in s.mutations:
            corrected = any(c.corrected and c.mutation_position == m.position for c in self._corrections)
            lines.append(f"  Mutation at {m.position} ({m.ref_base}->{m.alt_base}): {'CORRECTED' if corrected else 'NOT corrected'}")
        lines.append(f"Off-target damage: {len(self._off_target_damage)} site(s)")
        lines.append(f"Budget remaining: {s.experiment_budget - self._budget_used}")
        return "\n".join(lines), False

    def _handle_submit(self) -> Tuple[str, bool]:
        self._done = True
        score = self._compute_final_score()
        self._final_score = score
        return f"Solution submitted. Final score: {score:.4f}", False

    def _compute_final_score(self) -> float:
        s = self._scenario
        if s.task_type == "single_target":
            return grade_single_target(self._corrections, s.mutations, self._off_target_damage, self._budget_used, s.experiment_budget)
        elif s.task_type == "multi_repair":
            return grade_multi_repair(self._corrections, s.mutations, self._off_target_damage, self._budget_used, s.experiment_budget, self._edits_applied)
        elif s.task_type == "precision_editing":
            return grade_precision_editing(self._corrections, s.mutations, self._off_target_damage, s.regulatory_regions, self._budget_used, s.experiment_budget)
        return 0.0

    def _record_history(self, tool: str, args: str, output_summary: str) -> None:
        self._tool_history.append(ToolResultRecord(
            tool_name=tool, args=args, output_summary=output_summary[:200], step=self._steps_taken,
        ))
