from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Action, ActionType, EnvironmentState, GuideObservation, GuideRNA
from .reward import compute_reward
from .simulation import compute_corrected_mutations, success_probability
from .tasks import TASK_REGISTRY, TaskScenario


@dataclass
class EpisodeStats:
    cumulative_reward: float = 0.0
    corrected_positions: Optional[set[int]] = None


class CrisprEnv:
    """Gym-like environment for CRISPR optimization simulation."""

    def __init__(self, task_level: str = "easy", seed: int = 42):
        if task_level not in TASK_REGISTRY:
            raise ValueError(f"Unsupported task level: {task_level}")

        self.task_level = task_level
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self._scenario: Optional[TaskScenario] = None
        self._selected_guide: Optional[GuideRNA] = None
        self._steps_taken = 0
        self._done = False
        self._stats = EpisodeStats(cumulative_reward=0.0, corrected_positions=set())

    def reset(self) -> EnvironmentState:
        generator = TASK_REGISTRY[self.task_level]["generator"]
        self._scenario = generator(self._rng)
        self._selected_guide = None
        self._steps_taken = 0
        self._done = False
        self._stats = EpisodeStats(cumulative_reward=0.0, corrected_positions=set())
        return self.state()

    def state(self) -> EnvironmentState:
        if self._scenario is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        mutation_anchor = self._scenario.mutations[0].position
        seq = self._scenario.mutated_gene.sequence
        left = max(0, mutation_anchor - 12)
        right = min(len(seq), mutation_anchor + 13)
        sequence_window = seq[left:right]

        if self._scenario.noisy_observation:
            sequence_window = self._inject_window_noise(sequence_window)

        guides = [
            GuideObservation(
                id=guide.id,
                efficiency=guide.efficiency,
                off_target_risk=guide.off_target_risk,
                utility=float(np.clip(guide.efficiency - guide.off_target_risk, -1.0, 1.0)),
            )
            for guide in self._scenario.candidate_guides
        ]

        efficiency = self._selected_guide.efficiency if self._selected_guide else 0.0
        off_target = self._selected_guide.off_target_risk if self._selected_guide else 0.0

        return EnvironmentState(
            sequence_window=sequence_window,
            mutation_position=mutation_anchor,
            candidate_guides=guides,
            current_selected_guide=self._selected_guide.id if self._selected_guide else None,
            efficiency=efficiency,
            off_target_risk=off_target,
            steps_taken=self._steps_taken,
            corrected_mutations=len(self._stats.corrected_positions or set()),
            total_mutations=len(self._scenario.mutations),
        )

    def step(self, action: str) -> Tuple[EnvironmentState, float, bool, Dict]:
        if self._scenario is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")

        parsed_action = self._parse_action(action)
        self._steps_taken += 1
        info: Dict = {"action": parsed_action.action_type.value}

        if parsed_action.action_type == ActionType.SELECT_GUIDE:
            self._handle_select_guide(parsed_action.target_id)
            info["selected_guide"] = self._selected_guide.id if self._selected_guide else None

        elif parsed_action.action_type == ActionType.MODIFY_GUIDE:
            self._handle_modify_guide(parsed_action.modifier)
            info["modified"] = parsed_action.modifier

        elif parsed_action.action_type == ActionType.SIMULATE_EDIT:
            info["simulated_success_probability"] = self._current_success_probability()

        elif parsed_action.action_type == ActionType.APPLY_EDIT:
            corrected_now = self._handle_apply_edit()
            info["corrected_positions"] = sorted(corrected_now)

        elif parsed_action.action_type == ActionType.TERMINATE:
            self._done = True
            info["terminated"] = True

        if self._steps_taken >= self._scenario.max_steps:
            self._done = True
            info["max_steps_reached"] = True

        correctness = len(self._stats.corrected_positions or set()) / max(len(self._scenario.mutations), 1)
        efficiency = self._selected_guide.efficiency if self._selected_guide else 0.0
        off_target = self._selected_guide.off_target_risk if self._selected_guide else 1.0

        reward = compute_reward(
            correctness=correctness,
            efficiency=efficiency,
            off_target_risk=off_target,
            steps_taken=self._steps_taken,
            action_type=parsed_action.action_type.value,
            done=self._done,
        )
        self._stats.cumulative_reward += reward

        if self._done:
            grader = TASK_REGISTRY[self.task_level]["grader"]
            info["final_score"] = grader(
                len(self._stats.corrected_positions or set()),
                len(self._scenario.mutations),
                self._stats.cumulative_reward / max(self._steps_taken, 1),
            )

        return self.state(), reward, self._done, info

    def close(self) -> None:
        """Clean up the environment at the end of an episode."""
        self._scenario = None
        self._selected_guide = None
        self._done = True

    def _parse_action(self, action: str) -> Action:
        text = action.strip()

        if text.startswith("select_guide:"):
            _, guide_id = text.split(":", maxsplit=1)
            return Action(action_type=ActionType.SELECT_GUIDE, target_id=guide_id.strip())

        if text.startswith("modify_guide:"):
            _, modifier = text.split(":", maxsplit=1)
            return Action(action_type=ActionType.MODIFY_GUIDE, modifier=modifier.strip())

        if text == "simulate_edit":
            return Action(action_type=ActionType.SIMULATE_EDIT)

        if text == "apply_edit":
            return Action(action_type=ActionType.APPLY_EDIT)

        if text == "terminate":
            return Action(action_type=ActionType.TERMINATE)

        raise ValueError(f"Unsupported action: {action}")

    def _handle_select_guide(self, guide_id: Optional[str]) -> None:
        if not guide_id:
            raise ValueError("select_guide action requires a guide ID")

        for guide in self._scenario.candidate_guides:
            if guide.id == guide_id:
                self._selected_guide = guide
                return

        raise ValueError(f"Guide not found: {guide_id}")

    def _handle_modify_guide(self, modifier: Optional[str]) -> None:
        if self._selected_guide is None:
            raise ValueError("No selected guide to modify")

        if modifier != "increase_specificity":
            raise ValueError("Unsupported modifier. Use increase_specificity")

        self._selected_guide = GuideRNA(
            id=self._selected_guide.id,
            sequence=self._selected_guide.sequence,
            start=self._selected_guide.start,
            end=self._selected_guide.end,
            target_positions=self._selected_guide.target_positions,
            efficiency=float(np.clip(self._selected_guide.efficiency - 0.04, 0.0, 1.0)),
            off_target_risk=float(np.clip(self._selected_guide.off_target_risk - 0.12, 0.0, 1.0)),
        )

    def _current_success_probability(self) -> float:
        if self._selected_guide is None:
            return 0.0
        return success_probability(
            self._selected_guide.efficiency,
            self._selected_guide.off_target_risk,
        )

    def _handle_apply_edit(self) -> set[int]:
        if self._selected_guide is None:
            raise ValueError("Cannot apply edit without selecting a guide")

        corrected = set(
            compute_corrected_mutations(
                selected_guide=self._selected_guide,
                mutations=self._scenario.mutations,
                rng=self._rng,
            )
        )

        self._stats.corrected_positions = (self._stats.corrected_positions or set()).union(corrected)

        if len(self._stats.corrected_positions) == len(self._scenario.mutations):
            self._done = True

        return corrected

    def _inject_window_noise(self, sequence_window: str) -> str:
        window = list(sequence_window)
        for idx in range(len(window)):
            if float(self._rng.random()) < 0.04:
                alternatives = [b for b in ["A", "C", "G", "T"] if b != window[idx]]
                window[idx] = str(self._rng.choice(alternatives))
        return "".join(window)
