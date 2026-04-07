"""FastAPI server exposing the CrisprEnv via HTTP for HF Spaces / OpenEnv validation."""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import CrisprEnv

app = FastAPI(title="crispr-editing-env", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory environment store keyed by session_id (default: "default")
# ---------------------------------------------------------------------------
_envs: Dict[str, CrisprEnv] = {}


def _get_env(session_id: str = "default") -> CrisprEnv:
    if session_id not in _envs:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    return _envs[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_level: str = "easy"
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    action: str
    session_id: str = "default"


class StateRequest(BaseModel):
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "environment": "crispr-editing-env"}


@app.post("/reset")
def reset(req: ResetRequest):
    env = CrisprEnv(task_level=req.task_level, seed=req.seed)
    _envs[req.session_id] = env
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.session_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = _get_env(session_id)
    try:
        obs = env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"observation": obs.model_dump()}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "Single mutation with an obvious best guide."},
            {"name": "medium", "description": "Single mutation with multiple guide tradeoffs."},
            {"name": "hard", "description": "Multiple mutations with noisy observations and harder guide tradeoffs."},
        ]
    }
