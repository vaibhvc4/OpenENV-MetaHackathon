"""
Inference script for crispr-editing-env.

Uses the OpenAI client to run an LLM agent against all three CRISPR tasks
and emits structured [START]/[STEP]/[END] stdout logs.
"""

import json
import os
import sys
import traceback

from openai import OpenAI

from env.environment import CrisprEnv

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "crispr-editing-env"
MAX_STEPS = 10
TASKS = ["easy", "medium", "hard"]

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    return _client

# ---------------------------------------------------------------------------
# System prompt describing the environment to the LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an AI agent controlling a CRISPR gene-editing simulation.

Your goal: correct DNA mutations by selecting the best guide RNA, optionally \
improving its specificity, simulating the edit, and applying it.

Available actions (respond with EXACTLY one per turn):
  select_guide:<id>                 - Select a candidate guide RNA by its id
  modify_guide:increase_specificity - Improve specificity of the selected guide (lowers off-target risk)
  simulate_edit                     - Simulate the edit to preview success probability
  apply_edit                        - Apply the edit to correct mutations
  terminate                         - End the episode

Strategy tips:
- Pick the guide with the best trade-off of high efficiency and low off-target risk.
- Use modify_guide:increase_specificity if off-target risk is high.
- simulate_edit lets you check success probability before committing.
- apply_edit attempts to correct mutations. After applying, you can terminate.
- Respond with ONLY the action string, nothing else.
"""


def format_observation(obs) -> str:
    """Format an EnvironmentState into a concise text prompt for the LLM."""
    guides_text = "\n".join(
        f"  - {g.id}: efficiency={g.efficiency:.3f}, off_target_risk={g.off_target_risk:.3f}, utility={g.utility:.3f}"
        for g in obs.candidate_guides
    )
    return (
        f"Sequence window: {obs.sequence_window}\n"
        f"Mutation position: {obs.mutation_position}\n"
        f"Candidate guides:\n{guides_text}\n"
        f"Selected guide: {obs.current_selected_guide or 'None'}\n"
        f"Current efficiency: {obs.efficiency:.3f}\n"
        f"Off-target risk: {obs.off_target_risk:.3f}\n"
        f"Steps taken: {obs.steps_taken}/{MAX_STEPS}\n"
        f"Corrected mutations: {obs.corrected_mutations}/{obs.total_mutations}"
    )


def get_llm_action(messages: list) -> str:
    """Call the LLM and extract a single action string."""
    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    # Extract just the action line in case the model adds commentary
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("select_guide:"):
            return line
        if line.startswith("modify_guide:"):
            return line
        if line in ("simulate_edit", "apply_edit", "terminate"):
            return line
    # Fallback: return raw (will surface as an error in the step)
    return raw.splitlines()[0].strip() if raw else "terminate"


def run_task(task_name: str, seed: int = 42) -> float:
    """Run one episode of a task with the LLM agent, return final score."""
    env = CrisprEnv(task_level=task_name, seed=seed)
    obs = env.reset()

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]

    rewards = []
    done = False
    step_num = 0
    last_error = None
    final_score = 0.0

    try:
        while not done and step_num < MAX_STEPS:
            action = get_llm_action(messages)
            step_num += 1

            try:
                obs, reward, done, info = env.step(action)
                last_error = None
            except (ValueError, RuntimeError) as e:
                # Invalid action — record error, give zero reward, continue
                last_error = str(e)
                reward = 0.0
                done = False
                obs = env.state()

            rewards.append(reward)
            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={step_num} action={action} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}"
            )

            if not done:
                messages.append({"role": "assistant", "content": action})
                feedback = format_observation(obs)
                if last_error:
                    feedback = f"ERROR: {last_error}\n\n{feedback}"
                messages.append({"role": "user", "content": feedback})

        # Get final score from the last info dict if episode ended properly
        if done and "final_score" in info:
            final_score = info["final_score"]
        else:
            # Episode didn't finish via grader — compute a proxy score from rewards
            final_score = sum(rewards) / max(len(rewards), 1)
            final_score = max(0.0, min(1.0, final_score))

    except Exception:
        traceback.print_exc(file=sys.stderr)
        if not rewards:
            rewards = [0.0]
        final_score = 0.0

    success = final_score > 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )

    return final_score


def main():
    scores = {}
    for task_name in TASKS:
        score = run_task(task_name, seed=42)
        scores[task_name] = score
        print()  # blank line between tasks

    print("--- Summary ---")
    for task_name, score in scores.items():
        print(f"  {task_name}: {score:.2f}")
    overall = sum(scores.values()) / len(scores)
    print(f"  overall: {overall:.2f}")


if __name__ == "__main__":
    main()
