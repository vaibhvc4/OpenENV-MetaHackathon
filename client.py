"""Typed async client for the CRISPR v2 environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import CrisprAction, CrisprObservation, CrisprState


class CrisprEnvClient(EnvClient[CrisprAction, CrisprObservation, CrisprState]):
    """WebSocket client for interacting with a remote CrisprEnvironment server."""

    def _step_payload(self, action: CrisprAction) -> Dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CrisprObservation]:
        obs_data = payload.get("observation", payload)
        obs = CrisprObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
            metadata=obs.metadata,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CrisprState:
        return CrisprState(**payload)
