"""Pydantic models for the CRISPR v2 tool-based environment."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class MutationInfo(BaseModel):
    position: int = Field(..., ge=0)
    ref_base: str = Field(..., min_length=1, max_length=1)
    alt_base: str = Field(..., min_length=1, max_length=1)


class PAMSite(BaseModel):
    position: int
    strand: str  # "+" or "-"
    pattern: str  # e.g. "NGG"
    sequence: str  # actual bases at that position


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
    gc_quality: str  # "optimal" / "suboptimal" / "poor"
    homopolymer_max_run: int
    self_complementarity: float
    overall_quality: str  # "high" / "medium" / "low"


class OffTargetHit(BaseModel):
    position: int
    mismatches: int
    in_regulatory_region: bool
    risk_level: str  # "high" / "medium" / "low"


class CorrectionResult(BaseModel):
    mutation_position: int
    corrected: bool


class ToolResult(BaseModel):
    tool_name: str
    args: str
    output_summary: str
    step: int


class EnvironmentState(BaseModel):
    task_description: str
    task_type: str  # "single_target" / "multi_repair" / "precision_editing"
    target_gene_id: str
    target_gene_length: int
    known_mutations: List[MutationInfo]
    regulatory_regions: List[Tuple[int, int]] = Field(default_factory=list)
    experiment_budget: int
    max_steps: int
    steps_taken: int = 0
    edits_applied: int = 0
    last_tool: Optional[str] = None
    last_tool_output: Optional[str] = None
    last_tool_error: Optional[str] = None
    tool_history: List[ToolResult] = Field(default_factory=list)
    corrections_made: List[CorrectionResult] = Field(default_factory=list)
    off_target_damage: List[OffTargetHit] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=lambda: [
        "analyze_sequence <start> <end> — View GC content, repeats, structure for a region (FREE)",
        "search_pam_sites <pattern> — Find PAM motifs like NGG (FREE)",
        "design_guide <pam_position> <strand> — Design 20nt guide at a PAM site (FREE)",
        "evaluate_guide <guide_sequence> — Detailed quality scoring (1 credit)",
        "off_target_scan <guide_sequence> — Check off-target sites (2 credits)",
        "apply_edit <guide_sequence> <target_position> — Apply edit, irreversible (3 credits)",
        "check_edit_result — See corrections and damage so far (FREE)",
        "submit_solution — End episode, trigger grading (FREE)",
    ])
