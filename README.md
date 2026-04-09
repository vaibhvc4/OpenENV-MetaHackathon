---
title: Crispr Editing Env
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# crispr-editing-env

A production-quality OpenEnv environment that simulates **CRISPR gene editing optimization**. An AI agent must select and tune guide RNAs to correct DNA mutations while maximizing editing efficiency and minimizing off-target risk.

## Why CRISPR?

CRISPR guide RNA selection is a real-world bioinformatics task where researchers must balance multiple competing objectives (efficiency vs. safety) under uncertainty. This environment models that decision-making process as a sequential optimization problem suitable for RL and LLM agents.

## Action Space

| Action | Description |
|--------|-------------|
| `select_guide:<id>` | Select a candidate guide RNA by its ID |
| `modify_guide:increase_specificity` | Improve specificity of the selected guide (reduces off-target risk at slight efficiency cost) |
| `simulate_edit` | Preview the success probability before committing |
| `apply_edit` | Apply the edit to correct mutations |
| `terminate` | End the episode |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `sequence_window` | string | DNA sequence window around the mutation site |
| `mutation_position` | int | Position of the mutation in the full sequence |
| `candidate_guides` | list | Available guide RNAs with `id`, `efficiency`, `off_target_risk`, `utility` |
| `current_selected_guide` | string/null | Currently selected guide ID |
| `efficiency` | float [0,1] | Efficiency of the selected guide |
| `off_target_risk` | float [0,1] | Off-target risk of the selected guide |
| `steps_taken` | int | Steps taken in the current episode |
| `corrected_mutations` | int | Mutations successfully corrected |
| `total_mutations` | int | Total mutations to correct |

## Reward Function

Continuous reward in [0, 1] combining:
- **Correctness** (55%): fraction of mutations corrected
- **Efficiency** (30%): guide RNA efficiency score
- **Safety** (15%): penalty for off-target risk
- **Step penalty**: 2% per step (max 25%) to encourage efficiency
- **Action bonuses**: +2% for simulate, +5% for apply, -5% for early terminate

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `easy` | Low | Single mutation, one clearly best guide among 4 candidates | 6 |
| `medium` | Medium | Single mutation, 6 guides with competing tradeoffs | 8 |
| `hard` | High | 3 mutations, noisy observations, 5 guides per mutation | 10 |

Each task has a deterministic grader returning a score in [0.0, 1.0]:
- **Easy grader**: 80% accuracy + 20% reward
- **Medium grader**: 70% accuracy + 30% reward
- **Hard grader**: 60% accuracy + 40% reward

## Exploration visualization
<img width="1280" height="692" alt="Figure_1" src="https://github.com/user-attachments/assets/78ba2919-63c9-41b5-9b23-ce3d4062cd7d" />

## Baseline Scores (Rule-Based Agent)

```
seed=123, 5 episodes per task
----------------------------------------------------------
Task=  easy | avg_reward=2.1766 | avg_score=0.9088
Task=medium | avg_reward=1.9574 | avg_score=0.5561
Task=  hard | avg_reward=1.3852 | avg_score=0.1908
----------------------------------------------------------
Overall avg reward: 1.8397
Overall avg score : 0.5519
```

## Setup & Usage

### Install locally

```bash
pip install -r requirements.txt
python run_baseline.py          # Rule-based baseline
python inference.py             # LLM inference (needs HF_TOKEN)
```

### Docker

```bash
docker build -t crispr-editing-env .
docker run --rm -p 7860:7860 crispr-editing-env
```

### API Endpoints (when running as server)

```bash
# Health check
curl http://localhost:7860/

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "select_guide:g0"}'

# Get current state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

### Environment Variables for Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py
```

## Project Structure

```
crispr-editing-env/
├── env/
│   ├── environment.py      # CrisprEnv: step(), reset(), state()
│   ├── models.py           # Pydantic models (Observation, Action, Reward)
│   ├── simulation.py       # DNA/mutation/guide RNA simulation
│   ├── reward.py           # Reward computation
│   └── tasks.py            # Task definitions + graders
├── agents/
│   └── baseline.py         # Rule-based baseline agent
├── app.py                  # FastAPI server for HF Spaces
├── inference.py            # LLM inference with OpenAI client
├── run_baseline.py         # Local baseline runner
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile
└── requirements.txt
```
