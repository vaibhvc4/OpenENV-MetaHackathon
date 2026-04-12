"""
Inference script for crispr-editing-env v2.

Uses the OpenAI client to run an LLM agent with bioinformatics tools
against all three CRISPR tasks. Emits [START]/[STEP]/[END] stdout logs.
"""

import os
import sys
import traceback

from openai import OpenAI

from server.environment import CrisprEnvironment
from models import CrisprAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
if not API_KEY:
    raise RuntimeError("HF_TOKEN environment variable is required.")

BENCHMARK = "crispr-editing-env"
MAX_STEPS = 35
TASKS = ["single_target", "multi_repair", "precision_editing"]

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    return _client


SYSTEM_PROMPT = """\
You are a computational biologist using CRISPR-Cas9 to correct pathogenic DNA mutations.

You interact with a bioinformatics environment by issuing tool commands. Each turn, respond with EXACTLY ONE tool command (nothing else).

AVAILABLE TOOLS:
  analyze_sequence <start> <end>          — View GC content, repeats, structure for a gene region (FREE)
  search_pam_sites <pattern>              — Find PAM motifs like NGG or NNGRRT in the gene (FREE)
  design_guide <pam_position> <strand>    — Design a 20nt guide RNA at a PAM site (FREE)
  evaluate_guide <guide_sequence>         — Detailed quality scoring (costs 1 credit)
  off_target_scan <guide_sequence>        — Check for off-target binding sites (costs 2 credits)
  apply_edit <guide_sequence> <position>  — Apply the CRISPR edit, irreversible (costs 3 credits)
  check_edit_result                       — See which mutations are corrected and any damage (FREE)
  submit_solution                         — End the episode and get your final score (FREE)

STRATEGY:
1. Start by searching for PAM sites near the mutation(s): search_pam_sites NGG
2. Design guides at promising PAM sites (close to mutations, good strand)
3. Evaluate guide quality before committing credits
4. For hard tasks: ALWAYS run off_target_scan before apply_edit to check for regulatory region hits
5. Apply edit only when confident the guide is safe and effective
6. Submit when done or budget is low

IMPORTANT:
- Budget is LIMITED. Don't waste credits on evaluate/scan for every guide.
- apply_edit is IRREVERSIBLE and costs 3 credits. Plan carefully.
- If the task mentions a regulatory region, off-target damage there is catastrophic.
- Respond with ONLY the tool command. No explanation, no markdown, just the command.
"""


def format_observation(obs) -> str:
    """Format EnvironmentState as text for the LLM."""
    lines = [
        f"=== {obs.task_type} | Step {obs.steps_taken}/{obs.max_steps} | Budget: {obs.experiment_budget} credits ===",
        f"Gene: {obs.target_gene_id} ({obs.target_gene_length}bp)",
        f"Mutations to fix:",
    ]
    for m in obs.known_mutations:
        corrected = any(
            c.corrected and c.mutation_position == m.position
            for c in obs.corrections_made
        )
        status = " [CORRECTED]" if corrected else ""
        lines.append(f"  pos={m.position} {m.ref_base}->{m.alt_base}{status}")

    if obs.regulatory_regions:
        for rs, re in obs.regulatory_regions:
            lines.append(f"REGULATORY NO-EDIT ZONE: {rs}-{re} (DO NOT damage!)")

    lines.append(f"Edits applied: {obs.edits_applied}")
    if obs.off_target_damage:
        lines.append(f"Off-target damage: {len(obs.off_target_damage)} site(s)")

    if obs.last_tool_output:
        lines.append(f"\n--- Last tool output ({obs.last_tool}) ---")
        lines.append(obs.last_tool_output)
    elif obs.last_tool_error:
        lines.append(f"\n--- ERROR ({obs.last_tool}) ---")
        lines.append(obs.last_tool_error)

    return "\n".join(lines)


def get_llm_action(messages: list) -> str:
    """Call the LLM and extract a tool command."""
    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=128,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    # Extract the first valid-looking tool command
    valid_tools = [
        "analyze_sequence", "search_pam_sites", "design_guide",
        "evaluate_guide", "off_target_scan", "apply_edit",
        "check_edit_result", "submit_solution",
    ]
    for line in raw.splitlines():
        line = line.strip()
        if any(line.startswith(t) for t in valid_tools):
            return line
    # Fallback
    return raw.splitlines()[0].strip() if raw else "submit_solution"


def run_task(task_name: str, seed: int = 42) -> float:
    """Run one episode. Returns final score."""
    env = CrisprEnvironment(task_level=task_name, seed=seed)
    obs = env.reset(seed=seed)

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
            action_str = get_llm_action(messages)
            step_num += 1

            try:
                obs = env.step(CrisprAction(command=action_str))
                reward = obs.reward if obs.reward is not None else 0.0
                done = obs.done
                last_error = obs.last_tool_error
            except (ValueError, RuntimeError) as e:
                last_error = str(e)
                reward = 0.0
                done = False

            rewards.append(reward)
            error_str = last_error if last_error else "null"
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")

            if not done:
                messages.append({"role": "assistant", "content": action_str})
                feedback = format_observation(obs)
                if last_error:
                    feedback = f"ERROR: {last_error}\n\n{feedback}"
                messages.append({"role": "user", "content": feedback})

        if done and obs.metadata.get("final_score") is not None:
            final_score = obs.metadata["final_score"]
        else:
            final_score = sum(rewards) / max(len(rewards), 1)
            final_score = max(0.0, min(1.0, final_score))

    except Exception:
        traceback.print_exc(file=sys.stderr)
        if not rewards:
            rewards = [0.0]
        final_score = 0.0

    finally:
        env.close()
        success = final_score > 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={rewards_str}")

    return final_score


def main():
    for task_name in TASKS:
        run_task(task_name, seed=42)


if __name__ == "__main__":
    main()
