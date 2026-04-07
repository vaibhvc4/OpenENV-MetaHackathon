from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

DNA_ALPHABET = {"A", "C", "G", "T"}


class GeneSequence(BaseModel):
    id: str = Field(..., min_length=1)
    sequence: str = Field(..., min_length=1)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        seq = value.upper()
        if any(base not in DNA_ALPHABET for base in seq):
            raise ValueError("Gene sequence must contain only A/C/G/T")
        return seq


class Mutation(BaseModel):
    position: int = Field(..., ge=0)
    original_base: str = Field(..., min_length=1, max_length=1)
    mutated_base: str = Field(..., min_length=1, max_length=1)

    @field_validator("original_base", "mutated_base")
    @classmethod
    def validate_base(cls, value: str) -> str:
        base = value.upper()
        if base not in DNA_ALPHABET:
            raise ValueError("Mutation base must be one of A/C/G/T")
        return base


class GuideRNA(BaseModel):
    id: str = Field(..., min_length=1)
    sequence: str = Field(..., min_length=8)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)
    target_positions: List[int] = Field(default_factory=list)
    efficiency: float = Field(..., ge=0.0, le=1.0)
    off_target_risk: float = Field(..., ge=0.0, le=1.0)

    @field_validator("sequence")
    @classmethod
    def validate_guide_sequence(cls, value: str) -> str:
        seq = value.upper()
        if any(base not in DNA_ALPHABET for base in seq):
            raise ValueError("Guide sequence must contain only A/C/G/T")
        return seq

    @field_validator("end")
    @classmethod
    def validate_window(cls, value: int, info) -> int:
        start = info.data.get("start", 0)
        if value <= start:
            raise ValueError("Guide end must be greater than start")
        return value


class GuideObservation(BaseModel):
    id: str
    efficiency: float = Field(..., ge=0.0, le=1.0)
    off_target_risk: float = Field(..., ge=0.0, le=1.0)
    utility: float = Field(..., ge=-1.0, le=1.0)


class ActionType(str, Enum):
    SELECT_GUIDE = "select_guide"
    MODIFY_GUIDE = "modify_guide"
    SIMULATE_EDIT = "simulate_edit"
    APPLY_EDIT = "apply_edit"
    TERMINATE = "terminate"


class Action(BaseModel):
    action_type: ActionType
    target_id: Optional[str] = None
    modifier: Optional[str] = None


class EnvironmentState(BaseModel):
    sequence_window: str
    mutation_position: int
    candidate_guides: List[GuideObservation]
    current_selected_guide: Optional[str] = None
    efficiency: float = Field(..., ge=0.0, le=1.0)
    off_target_risk: float = Field(..., ge=0.0, le=1.0)
    steps_taken: int = Field(..., ge=0)
    corrected_mutations: int = Field(..., ge=0)
    total_mutations: int = Field(..., ge=1)
