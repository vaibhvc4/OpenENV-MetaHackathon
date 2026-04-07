"""
Manual step-by-step execution of CRISPR environment.
Run this to interactively explore the environment and see state changes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from env.environment import CrisprEnv


def print_state(obs, title: str = "State") -> None:
    """Pretty-print the environment state."""
    print(f"\n{title}:")
    print(f"  Sequence Window: {obs.sequence_window}")
    print(f"  Mutation Position: {obs.mutation_position}")
    print(f"  Steps Taken: {obs.steps_taken}")
    print(f"  Corrected/Total Mutations: {obs.corrected_mutations}/{obs.total_mutations}")
    print(f"  Current Selected Guide: {obs.current_selected_guide}")
    print(f"  Efficiency: {obs.efficiency:.4f}")
    print(f"  Off-target Risk: {obs.off_target_risk:.4f}")
    print(f"\n  Available Guides ({len(obs.candidate_guides)}):")
    for guide in obs.candidate_guides:
        utility = guide.efficiency - guide.off_target_risk
        print(f"    {guide.id}: efficiency={guide.efficiency:.3f}, off_target={guide.off_target_risk:.3f}, utility={utility:.3f}")


def plot_charts(
    step_labels: list[str],
    efficiencies: list[float],
    off_target_risks: list[float],
    rewards: list[float],
    corrected: list[int],
    total_mutations: int,
    initial_guides: list,
) -> None:
    """Render episode charts using matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CRISPR Editing Episode – Manual Exploration", fontsize=14, fontweight="bold")

    steps = range(len(step_labels))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Chart 1: Efficiency & Off-target Risk over steps ---
    ax1 = axes[0, 0]
    ax1.plot(steps, efficiencies, marker="o", label="Efficiency", color=colors[0])
    ax1.plot(steps, off_target_risks, marker="s", linestyle="--", label="Off-target Risk", color=colors[1])
    ax1.set_title("Efficiency & Off-target Risk per Step")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Value")
    ax1.set_xticks(list(steps))
    ax1.set_xticklabels(step_labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Chart 2: Reward per step ---
    ax2 = axes[0, 1]
    bar_colors = [colors[2] if r >= 0 else colors[3] for r in rewards]
    ax2.bar(list(steps), rewards, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Reward per Step")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.set_xticks(list(steps))
    ax2.set_xticklabels(step_labels, rotation=20, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Chart 3: Mutation correction progress ---
    ax3 = axes[1, 0]
    ax3.step(list(steps), corrected, where="post", color=colors[4], linewidth=2)
    ax3.fill_between(list(steps), corrected, step="post", alpha=0.2, color=colors[4])
    ax3.axhline(total_mutations, color="red", linestyle=":", linewidth=1.2, label=f"Total mutations ({total_mutations})")
    ax3.set_title("Corrected Mutations over Steps")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Mutations Corrected")
    ax3.set_xticks(list(steps))
    ax3.set_xticklabels(step_labels, rotation=20, ha="right", fontsize=8)
    ax3.set_ylim(-0.1, max(total_mutations, 1) + 0.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- Chart 4: Guide comparison (initial candidates) ---
    ax4 = axes[1, 1]
    if initial_guides:
        guide_ids = [g.id for g in initial_guides]
        guide_eff = [g.efficiency for g in initial_guides]
        guide_ot = [g.off_target_risk for g in initial_guides]
        guide_utility = [g.efficiency - g.off_target_risk for g in initial_guides]
        x = np.arange(len(guide_ids))
        width = 0.28
        ax4.bar(x - width, guide_eff, width, label="Efficiency", color=colors[0])
        ax4.bar(x, guide_ot, width, label="Off-target Risk", color=colors[1])
        ax4.bar(x + width, guide_utility, width, label="Utility (eff−ot)", color=colors[2])
        ax4.set_title("Initial Candidate Guide Comparison")
        ax4.set_xlabel("Guide")
        ax4.set_ylabel("Value")
        ax4.set_xticks(x)
        ax4.set_xticklabels(guide_ids, fontsize=8)
        ax4.axhline(0, color="black", linewidth=0.6)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")
    else:
        ax4.text(0.5, 0.5, "No guide data", ha="center", va="center", transform=ax4.transAxes)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Manual exploration of the CRISPR environment."""
    print("=" * 70)
    print("CRISPR Editing Environment - Manual Exploration".center(70))
    print("=" * 70)

    # Initialize environment
    print("\n[1] Initializing environment with task_level='easy', seed=42")
    env = CrisprEnv(task_level="easy", seed=42)

    # Reset and get initial state
    print("[2] Calling reset() to initialize new mutation scenario")
    obs = env.reset()
    print_state(obs, "Initial State")

    # Capture initial guides for chart 4
    initial_guides = list(obs.candidate_guides)
    total_mutations = obs.total_mutations

    # Tracking lists (post-reset baseline)
    step_labels: list[str] = ["reset"]
    efficiencies: list[float] = [obs.efficiency]
    off_target_risks: list[float] = [obs.off_target_risk]
    rewards: list[float] = [0.0]
    corrected: list[int] = [obs.corrected_mutations]

    def _record(label: str, obs, reward: float) -> None:
        step_labels.append(label)
        efficiencies.append(obs.efficiency)
        off_target_risks.append(obs.off_target_risk)
        rewards.append(reward)
        corrected.append(obs.corrected_mutations)

    # Step 1: Select a guide
    print("\n[3] Action: select_guide:g0 (select first guide)")
    action1 = "select_guide:g0"
    obs, reward, done, info = env.step(action1)
    print(f"  Reward: {reward:.4f}, Done: {done}")
    print(f"  Info: {info}")
    print_state(obs, "State After Select")
    _record("select_guide", obs, reward)

    # Step 2: Modify guide for better specificity
    print("\n[4] Action: modify_guide:increase_specificity (reduce off-target risk)")
    action2 = "modify_guide:increase_specificity"
    obs, reward, done, info = env.step(action2)
    print(f"  Reward: {reward:.4f}, Done: {done}")
    print(f"  Info: {info}")
    print_state(obs, "State After Modify")
    _record("modify_guide", obs, reward)

    # Step 3: Simulate edit
    print("\n[5] Action: simulate_edit (check success probability)")
    action3 = "simulate_edit"
    obs, reward, done, info = env.step(action3)
    print(f"  Reward: {reward:.4f}, Done: {done}")
    print(f"  Simulated Success Probability: {info.get('simulated_success_probability', 'N/A'):.4f}")
    print(f"  Info: {info}")
    print_state(obs, "State After Simulate")
    _record("simulate_edit", obs, reward)

    # Step 4: Apply edit
    print("\n[6] Action: apply_edit (execute the edit)")
    action4 = "apply_edit"
    obs, reward, done, info = env.step(action4)
    print(f"  Reward: {reward:.4f}, Done: {done}")
    print(f"  Corrected Positions: {info.get('corrected_positions', [])}")
    print(f"  Info: {info}")
    print_state(obs, "State After Apply")
    _record("apply_edit", obs, reward)

    # Terminate (if episode not already done)
    if not done:
        print("\n[7] Action: terminate (end episode)")
        action5 = "terminate"
        obs, reward, done, info = env.step(action5)
        print(f"  Reward: {reward:.4f}, Done: {done}")
        print(f"  Final Score: {info.get('final_score', 'N/A'):.4f}")
        print(f"  Info: {info}")
        print_state(obs, "Final State")
        _record("terminate", obs, reward)
    else:
        print("\n[7] Episode already ended (all mutations corrected)!")
        print(f"  Final Score: {info.get('final_score', 'N/A'):.4f}")

    print("\n" + "=" * 70)
    print("Episode Complete".center(70))
    print("=" * 70)

    # Render charts
    print("\n[8] Rendering episode charts...")
    plot_charts(
        step_labels=step_labels,
        efficiencies=efficiencies,
        off_target_risks=off_target_risks,
        rewards=rewards,
        corrected=corrected,
        total_mutations=total_mutations,
        initial_guides=initial_guides,
    )

    # Show how to try different tasks
    print("\n\nTo explore other tasks, you can:")
    print("  1. Change task_level: 'easy', 'medium', or 'hard'")
    print("  2. Change seed: any integer for reproducibility")
    print("\nExample:")
    print("  env = CrisprEnv(task_level='medium', seed=999)")
    print("  obs = env.reset()")
    print("  obs, reward, done, info = env.step('select_guide:g0')")
    print("\nAvailable actions:")
    print("  - select_guide:<id>")
    print("  - modify_guide:increase_specificity")
    print("  - simulate_edit")
    print("  - apply_edit")
    print("  - terminate")


if __name__ == "__main__":
    main()
