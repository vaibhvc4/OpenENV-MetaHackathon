"""Step-level reward computation and tool cost tracking."""

from __future__ import annotations

import numpy as np

TOOL_COSTS = {
    "analyze_sequence": 0,
    "search_pam_sites": 0,
    "design_guide": 0,
    "evaluate_guide": 1,
    "off_target_scan": 2,
    "apply_edit": 3,
    "check_edit_result": 0,
    "submit_solution": 0,
}


def compute_step_reward(
    tool_name: str,
    tool_succeeded: bool,
    explored_new_info: bool,
) -> float:
    """Small per-step reward that encourages exploration without dominating final score."""
    if not tool_succeeded:
        return -0.01

    reward = 0.0

    # Free info-gathering tools get small reward for new info
    if tool_name in ("analyze_sequence", "search_pam_sites", "design_guide"):
        reward += 0.02 if explored_new_info else 0.005

    # Paid tools: no step reward (value deferred to final grade)
    elif tool_name in ("evaluate_guide", "off_target_scan", "apply_edit"):
        reward += 0.01

    # Notebook / check: encourage
    elif tool_name == "check_edit_result":
        reward += 0.01

    # Submit: no step reward
    elif tool_name == "submit_solution":
        reward += 0.0

    # Small step cost to encourage efficiency
    reward -= 0.005

    return float(np.clip(reward, -0.05, 0.1))
