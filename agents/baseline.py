from __future__ import annotations

from typing import List, Tuple

from env.environment import CrisprEnv


class BaselineAgent:
    """Rule-based baseline: choose max (efficiency - off_target), then apply edit."""

    def choose_best_guide_id(self, observation) -> str:
        guides = observation.candidate_guides
        best = max(guides, key=lambda g: g.efficiency - g.off_target_risk)
        return best.id

    def run_episode(self, env: CrisprEnv, allow_modify: bool = True) -> Tuple[float, dict, List[str]]:
        obs = env.reset()
        total_reward = 0.0
        action_trace: List[str] = []
        done = False
        info = {}

        best_guide_id = self.choose_best_guide_id(obs)
        planned_actions = [f"select_guide:{best_guide_id}"]

        if allow_modify:
            planned_actions.append("modify_guide:increase_specificity")

        planned_actions.extend(["simulate_edit", "apply_edit", "terminate"])

        for action in planned_actions:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            action_trace.append(action)
            if done:
                break

        return total_reward, info, action_trace
