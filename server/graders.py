"""Final episode grading for each task type."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .models import CorrectionResult, MutationInfo, OffTargetHit


def grade_single_target(
    corrections: List[CorrectionResult],
    mutations: List[MutationInfo],
    off_target_damage: List[OffTargetHit],
    budget_used: int,
    total_budget: int,
) -> float:
    """Easy task grading: 70% correction, 20% safety, 10% efficiency."""
    total = max(len(mutations), 1)
    corrected = sum(1 for c in corrections if c.corrected)
    correction_score = corrected / total

    damage_count = len(off_target_damage)
    safety_score = max(0.0, 1.0 - damage_count * 0.3)

    efficiency = 1.0 - (budget_used / max(total_budget, 1))

    score = 0.70 * correction_score + 0.20 * safety_score + 0.10 * efficiency
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def grade_multi_repair(
    corrections: List[CorrectionResult],
    mutations: List[MutationInfo],
    off_target_damage: List[OffTargetHit],
    budget_used: int,
    total_budget: int,
    edits_applied: int,
) -> float:
    """Medium task grading: 50% correction, 25% safety, 15% efficiency, 10% grouping."""
    total = max(len(mutations), 1)
    corrected = sum(1 for c in corrections if c.corrected)
    correction_score = corrected / total

    damage_count = len(off_target_damage)
    safety_score = max(0.0, 1.0 - damage_count * 0.25)

    efficiency = 1.0 - (budget_used / max(total_budget, 1))

    # Grouping bonus: fewer edits to fix more mutations is better
    if corrected > 0 and edits_applied > 0:
        grouping_bonus = min(1.0, corrected / edits_applied) * 0.5 + 0.5
    else:
        grouping_bonus = 0.0

    score = (0.50 * correction_score + 0.25 * safety_score +
             0.15 * efficiency + 0.10 * grouping_bonus)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def grade_precision_editing(
    corrections: List[CorrectionResult],
    mutations: List[MutationInfo],
    off_target_damage: List[OffTargetHit],
    regulatory_regions: List[Tuple[int, int]],
    budget_used: int,
    total_budget: int,
) -> float:
    """Hard task grading: 35% correction, 25% safety, 30% regulatory integrity, 10% efficiency.
    Any off-target damage in the regulatory region is catastrophic (zone_score = 0)."""
    total = max(len(mutations), 1)
    corrected = sum(1 for c in corrections if c.corrected)
    correction_score = corrected / total

    # General safety
    non_reg_damage = [d for d in off_target_damage if not d.in_regulatory_region]
    safety_score = max(0.0, 1.0 - len(non_reg_damage) * 0.25)

    # Regulatory zone integrity — binary: any damage = catastrophic
    reg_damage = [d for d in off_target_damage if d.in_regulatory_region]
    zone_score = 0.0 if reg_damage else 1.0

    efficiency = 1.0 - (budget_used / max(total_budget, 1))

    score = (0.35 * correction_score + 0.25 * safety_score +
             0.30 * zone_score + 0.10 * efficiency)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


GRADER_REGISTRY = {
    "single_target": grade_single_target,
    "multi_repair": grade_multi_repair,
    "precision_editing": grade_precision_editing,
}
