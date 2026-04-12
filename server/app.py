"""FastAPI server for CRISPR v2 — uses openenv create_app() factory."""

from __future__ import annotations

import os

import uvicorn
from openenv.core import create_app

from .environment import CrisprEnvironment
from .models import CrisprAction, CrisprObservation

TASK_LEVEL = os.getenv("CRISPR_TASK", "single_target")


def create_environment() -> CrisprEnvironment:
    return CrisprEnvironment(task_level=TASK_LEVEL)


app = create_app(
    create_environment,
    CrisprAction,
    CrisprObservation,
    env_name="crispr-editing-env",
    max_concurrent_envs=1,
)


def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app="server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
