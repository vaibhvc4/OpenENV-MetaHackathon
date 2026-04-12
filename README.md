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

# crispr-editing-env v2

A tool-based OpenEnv environment for **CRISPR gene editing**. An AI agent uses bioinformatics tools to analyze DNA sequences, find PAM sites, design guide RNAs, evaluate off-target risks, and apply precision edits under resource constraints.

## Why CRISPR?

CRISPR guide RNA design is a real-world bioinformatics task requiring multi-step investigation: researchers must find PAM sites, design guides, evaluate safety, and manage limited experimental resources. This environment models that workflow as a sequential tool-use problem for LLM agents.

## Action Space (8 Bioinformatics Tools)

| Tool | Description | Cost |
|------|-------------|------|
| `analyze_sequence <start> <end>` | View GC content, repeats, structure for a gene region | FREE |
| `search_pam_sites <pattern>` | Find PAM motifs (NGG, NNGRRT) in the gene | FREE |
| `design_guide <pam_pos> <strand>` | Design a 20nt guide RNA at a PAM site | FREE |
| `evaluate_guide <sequence>` | Detailed quality scoring | 1 credit |
| `off_target_scan <sequence>` | Check for off-target binding sites | 2 credits |
| `apply_edit <sequence> <position>` | Apply the edit (irreversible) | 3 credits |
| `check_edit_result` | See corrections and damage so far | FREE |
| `submit_solution` | End episode, trigger grading | FREE |

The agent sends free-form tool commands as actions. Budget limits prevent brute-force approaches.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | string | What the agent must accomplish |
| `task_type` | string | single_target / multi_repair / precision_editing |
| `target_gene_id` | string | Gene identifier |
| `target_gene_length` | int | Gene length in base pairs |
| `known_mutations` | list | Mutations to correct (position, ref_base, alt_base) |
| `regulatory_regions` | list | No-edit zones (hard task only) |
| `experiment_budget` | int | Remaining experiment credits |
| `last_tool_output` | string | Raw output from the last tool call |
| `tool_history` | list | Previous tool calls and results |
| `corrections_made` | list | Which mutations have been corrected |
| `off_target_damage` | list | Any collateral damage from edits |

No pre-computed guide candidates or utility scores. The agent must discover everything through tool use.

## Tasks (3 Qualitatively Different)

| Task | Difficulty | Description | Budget | Max Steps |
|------|-----------|-------------|--------|-----------|
| `single_target` | Easy | Correct one mutation. Find PAM sites, design guide, evaluate, apply. | 20 | 25 |
| `multi_repair` | Medium | Correct 3 mutations with limited budget. Must plan which to group. | 15 | 30 |
| `precision_editing` | Hard | Correct 2 mutations near a regulatory no-edit zone. Obvious guide is a trap. | 12 | 35 |

### Grading

- **Easy**: 70% correction + 20% safety + 10% efficiency
- **Medium**: 50% correction + 25% safety + 15% efficiency + 10% grouping bonus
- **Hard**: 35% correction + 25% safety + 30% regulatory integrity (binary!) + 10% efficiency

## Baseline Scores (Greedy Agent, 10 seeds)

```
single_target       : avg=0.77
multi_repair        : avg=0.63
precision_editing   : avg=0.49
```

The greedy baseline searches for the nearest PAM site and applies without evaluating off-targets. On the hard task, this frequently damages the regulatory region.

## Setup & Usage

### Install locally

```bash
pip install -r requirements.txt
python inference.py             # LLM inference (needs HF_TOKEN)
```

### Docker

```bash
docker build -t crispr-editing-env .
docker run --rm -p 7860:7860 crispr-editing-env
```

### API Endpoints

```bash
# Health check
curl http://localhost:7860/

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "single_target", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "search_pam_sites NGG"}'

# Get current state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

### Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py
```

## Project Structure

```
crispr-editing-env/
├── server/
│   ├── app.py              # FastAPI server (reset/step/state/tasks)
│   ├── environment.py      # CrisprEnv: tool dispatch engine
│   ├── models.py           # Pydantic models
│   ├── simulation.py       # PAM search, guide design, off-target scan
│   ├── tasks.py            # 3 task generators
│   ├── graders.py          # Final scoring logic
│   └── reward.py           # Step-level rewards + tool costs
├── app.py                  # Root re-export for Dockerfile
├── inference.py            # LLM agent with [START]/[STEP]/[END] format
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```
