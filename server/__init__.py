"""CRISPR v2 server package."""

from .models import (
    CrisprAction,
    CrisprObservation,
    CrisprState,
    CorrectionResult,
    GuideDesign,
    GuideEvaluation,
    MutationInfo,
    OffTargetHit,
    ToolResultRecord,
)

__all__ = [
    "CrisprEnvironment",
    "CrisprAction",
    "CrisprObservation",
    "CrisprState",
    "CorrectionResult",
    "GuideDesign",
    "GuideEvaluation",
    "MutationInfo",
    "OffTargetHit",
    "ToolResultRecord",
]


def __getattr__(name):
    if name == "CrisprEnvironment":
        from .environment import CrisprEnvironment
        return CrisprEnvironment
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
