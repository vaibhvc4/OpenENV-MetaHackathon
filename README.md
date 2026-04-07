# crispr-editing-env

A production-quality, lightweight OpenEnv environment that simulates CRISPR gene editing optimization.

The environment models a mutated DNA sequence, candidate guide RNAs, and editing actions. An agent must choose and adjust guides to maximize edit correctness and efficiency while minimizing off-target risk.

## Environment Overview

`CrisprEnv` uses a Gym-like API:

- `reset()` -> creates a new mutation scenario and returns the initial observation
- `step(action)` -> applies action and returns `(state, reward, done, info)`
- `state()` -> returns the current observation

Simulation features:

- Random DNA sequence generation
- Substitution mutation injection
- Candidate guide RNA generation around mutation windows
- Efficiency score in `[0, 1]`
- Off-target risk in `[0, 1]`
- Edit success probability: `success_probability = efficiency - off_target_risk` (clipped to `[0,1]`)
- Reproducibility through fixed random seeds

## Action Space

Supported actions:

- `select_guide:<id>`
- `modify_guide:increase_specificity`
- `simulate_edit`
- `apply_edit`
- `terminate`

## Observation Space

Each observation contains:

- `sequence_window` (string)
- `mutation_position` (int)
- `candidate_guides` (list with `id`, `efficiency`, `off_target_risk`, `utility`)
- `current_selected_guide` (string or null)
- `efficiency` (float)
- `off_target_risk` (float)
- `steps_taken` (int)
- `corrected_mutations` (int)
- `total_mutations` (int)

## Reward Function

Continuous reward in `[0, 1]`, combining:

- correctness of edit
- guide efficiency
- off-target penalty
- per-step penalty

The environment provides partial rewards at each step and final task scores via task-specific graders.

## Tasks

Three task levels are included:

- Easy: one mutation, clear best guide
- Medium: one mutation, guide tradeoffs
- Hard: multiple mutations + noisy observation window

Each task has:

- a scenario generator
- a grader returning score `[0, 1]`

## Baseline Agent

`agents/baseline.py` implements a rule-based policy:

- chooses guide maximizing `efficiency - off_target_risk`
- optionally applies specificity modification
- simulates and applies edit
- terminates episode

## Run Locally

From this directory:

```powershell
python -m pip install -r requirements.txt
python run_baseline.py
```

Expected output pattern:

```text
Running baseline with fixed seed=123
--------------------------------------------------------
Task=  easy | avg_reward=... | avg_score=...
Task=medium | avg_reward=... | avg_score=...
Task=  hard | avg_reward=... | avg_score=...
--------------------------------------------------------
Overall avg reward: ...
Overall avg score : ...
```

## Run with Docker

Build and run:

```powershell
docker build -t crispr-editing-env .
docker run --rm crispr-editing-env
```

## Hugging Face Spaces Readiness

- CPU-only, no GPU dependencies
- script-first entrypoint (`run_baseline.py`)
- fast startup with minimal dependencies (`numpy`, `pydantic`)

## File Layout

```text
crispr-editing-env/
├── env/
│   ├── environment.py
│   ├── models.py
│   ├── simulation.py
│   ├── reward.py
│   ├── tasks.py
├── agents/
│   ├── baseline.py
├── run_baseline.py
├── openenv.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```
---
title: Crispr Editing Env
emoji: 📚
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
