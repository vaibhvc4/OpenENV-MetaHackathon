"""Pydantic models for the CRISPR v2 tool-based environment.

Extends openenv-core base classes: Action, Observation, State.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openenv.core import Action, Observation, State
from pydantic import BaseModel, Field


class MutationInfo(BaseModel):
    position: int = Field(..., ge=0)
    ref_base: str = Field(..., min_length=1, max_length=1)
    alt_base: str = Field(..., min_length=1, max_length=1)


class PAMSite(BaseModel):
    position: int
    strand: str
    pattern: str
    sequence: str


class GuideDesign(BaseModel):
    guide_id: str
    sequence: str
    pam_position: int
    strand: str
    distance_to_nearest_mutation: int
    gc_content: float
    on_target_score: float


class GuideEvaluation(BaseModel):
    guide_id: str
    sequence: str
    gc_content: float
    gc_quality: str
    homopolymer_max_run: int
    self_complementarity: float
    overall_quality: str


class OffTargetHit(BaseModel):
    position: int
    mismatches: int
    in_regulatory_region: bool
    risk_level: str


class CorrectionResult(BaseModel):
    mutation_position: int
    corrected: bool


class ToolResultRecord(BaseModel):
    tool_name: str
    args: str
    output_summary: str
    step: int


# ── OpenEnv Action ───────────────────────────────────────────────────────

class CrisprAction(Action):
    """Agent's action: a tool command string."""
    command: str = Field(..., description="Tool command, e.g. 'search_pam_sites NGG'")


# ── OpenEnv Observation ──────────────────────────────────────────────────

class CrisprObservation(Observation):
    """Observation returned after each step."""
    task_description: str = ""
    task_type: str = ""
    target_gene_id: str = ""
    target_gene_length: int = 0
    known_mutations: List[MutationInfo] = Field(default_factory=list)
    regulatory_regions: List[Tuple[int, int]] = Field(default_factory=list)
    experiment_budget: int = 0
    max_steps: int = 0
    steps_taken: int = 0
    edits_applied: int = 0
    last_tool: Optional[str] = None
    last_tool_output: Optional[str] = None
    last_tool_error: Optional[str] = None
    tool_history: List[ToolResultRecord] = Field(default_factory=list)
    corrections_made: List[CorrectionResult] = Field(default_factory=list)
    off_target_damage: List[OffTargetHit] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=lambda: [
        "analyze_sequence <start> <end> — View GC content, repeats, structure (FREE)",
        "search_pam_sites <pattern> — Find PAM motifs like NGG (FREE)",
        "design_guide <pam_position> <strand> — Design 20nt guide at PAM site (FREE)",
        "evaluate_guide <guide_sequence> — Quality scoring (1 credit)",
        "off_target_scan <guide_sequence> — Off-target check (2 credits)",
        "apply_edit <guide_sequence> <target_position> — Apply edit (3 credits)",
        "check_edit_result — See corrections and damage (FREE)",
        "submit_solution — End episode, trigger grading (FREE)",
    ])


# ── OpenEnv State ────────────────────────────────────────────────────────

class CrisprState(State):
    """Server-side state for the environment."""
    task_type: str = ""
    target_gene_id: str = ""
    budget_remaining: int = 0
    edits_applied: int = 0
    corrections_count: int = 0
    damage_count: int = 0
