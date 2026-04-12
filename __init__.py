"""CRISPR Editing Environment — OpenEnv compliant bioinformatics tool-use environment."""

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
    "CrisprEnvClient",
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
    if name == "CrisprEnvClient":
        from .client import CrisprEnvClient
        return CrisprEnvClient
    if name == "CrisprEnvironment":
        from .server.environment import CrisprEnvironment
        return CrisprEnvironment
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
