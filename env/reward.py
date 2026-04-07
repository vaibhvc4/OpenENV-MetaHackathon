from __future__ import annotations

import numpy as np


def compute_reward(
    correctness: float,
    efficiency: float,
    off_target_risk: float,
    steps_taken: int,
    action_type: str,
    done: bool,
) -> float:
    """Continuous reward in [0, 1] with partial rewards each step."""
    correctness = float(np.clip(correctness, 0.0, 1.0))
    efficiency = float(np.clip(efficiency, 0.0, 1.0))
    off_target_risk = float(np.clip(off_target_risk, 0.0, 1.0))

    base = 0.55 * correctness + 0.3 * efficiency + 0.15 * (1.0 - off_target_risk)
    step_penalty = min(0.02 * steps_taken, 0.25)

    action_bonus = 0.0
    if action_type == "simulate_edit":
        action_bonus = 0.02
    elif action_type == "apply_edit":
        action_bonus = 0.05
    elif action_type == "terminate" and not done:
        action_bonus = -0.05

    reward = base + action_bonus - step_penalty
    return float(np.clip(reward, 0.0, 1.0))
