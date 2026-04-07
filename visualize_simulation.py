from __future__ import annotations

from statistics import mean, stdev
from typing import List

from agents.baseline import BaselineAgent
from env.environment import CrisprEnv


def format_header(text: str, width: int = 70) -> str:
    """Format a centered header with borders."""
    return f"\n{'=' * width}\n{text.center(width)}\n{'=' * width}\n"


def format_table_row(columns: List[str], widths: List[int]) -> str:
    """Format a table row with specified column widths."""
    return " | ".join(
        str(col).ljust(w) if i == 0 else str(col).rjust(w)
        for i, (col, w) in enumerate(zip(columns, widths))
    )


def format_separator(widths: List[int]) -> str:
    """Format a table separator line."""
    return "-+-".join("-" * w for w in widths)


def print_episode_detail(task_name: str, episode: int, reward: float, score: float, info: dict, trace: List[str]) -> None:
    """Print detailed information about a single episode."""
    print(f"\n  Episode {episode + 1}:")
    print(f"    Reward: {reward:.4f} | Score: {score:.4f}")
    print(f"    Corrected: {info.get('corrected_positions', [])} | Max Steps: {info.get('max_steps_reached', False)}")
    print(f"    Actions: {' → '.join(trace)}")


def run_task_with_details(task_name: str, seed: int, episodes: int = 5) -> tuple[List[float], List[float]]:
    """Run a task and collect episode details."""
    rewards = []
    scores = []
    agent = BaselineAgent()

    print(f"\n  Running {episodes} episodes...")

    for episode in range(episodes):
        env = CrisprEnv(task_level=task_name, seed=seed + episode)
        total_reward, info, trace = agent.run_episode(env)
        rewards.append(total_reward)
        scores.append(float(info.get("final_score", 0.0)))
        print_episode_detail(task_name, episode, total_reward, scores[-1], info, trace)

    return rewards, scores


def print_task_summary(task_name: str, rewards: List[float], scores: List[float]) -> tuple[float, float]:
    """Print summary statistics for a task."""
    avg_reward = mean(rewards)
    avg_score = mean(scores)
    std_reward = stdev(rewards) if len(rewards) > 1 else 0.0
    std_score = stdev(scores) if len(scores) > 1 else 0.0

    print(f"\n  {task_name.upper()} Task Summary:")
    print(f"    Avg Reward: {avg_reward:.4f} (±{std_reward:.4f})")
    print(f"    Avg Score:  {avg_score:.4f} (±{std_score:.4f})")
    print(f"    Min/Max Reward: [{min(rewards):.4f}, {max(rewards):.4f}]")
    print(f"    Min/Max Score:  [{min(scores):.4f}, {max(scores):.4f}]")

    return avg_reward, avg_score


def main() -> None:
    seed = 123
    tasks = ["easy", "medium", "hard"]

    print(format_header("CRISPR Editing Environment - Baseline Visualization", 70))
    print(f"Configuration: seed={seed}, episodes=5 per task")

    all_task_rewards = []
    all_task_scores = []
    task_summaries = []

    for task in tasks:
        print(format_header(f"Task: {task.upper()}", 70))
        rewards, scores = run_task_with_details(task_name=task, seed=seed, episodes=5)
        avg_reward, avg_score = print_task_summary(task, rewards, scores)

        all_task_rewards.extend(rewards)
        all_task_scores.extend(scores)
        task_summaries.append((task.upper(), avg_reward, avg_score, len(rewards)))

    # Final summary table
    print(format_header("FINAL RESULTS SUMMARY", 70))

    headers = ["Task", "Avg Reward", "Avg Score", "Episodes"]
    widths = [12, 15, 15, 12]

    print(format_table_row(headers, widths))
    print(format_separator(widths))

    for task_name, avg_reward, avg_score, num_episodes in task_summaries:
        row = [task_name, f"{avg_reward:.4f}", f"{avg_score:.4f}", str(num_episodes)]
        print(format_table_row(row, widths))

    print("\n" + format_separator(widths))

    overall_avg_reward = mean(all_task_rewards)
    overall_avg_score = mean(all_task_scores)
    overall_std_reward = stdev(all_task_rewards)
    overall_std_score = stdev(all_task_scores)

    row = ["OVERALL", f"{overall_avg_reward:.4f}", f"{overall_avg_score:.4f}", str(len(all_task_rewards))]
    print(format_table_row(row, widths))

    print(f"\n\nOverall Statistics:")
    print(f"  Total Episodes: {len(all_task_rewards)}")
    print(f"  Average Reward: {overall_avg_reward:.4f} (±{overall_std_reward:.4f})")
    print(f"  Average Score:  {overall_avg_score:.4f} (±{overall_std_score:.4f})")
    print(f"  Reward Range:   [{min(all_task_rewards):.4f}, {max(all_task_rewards):.4f}]")
    print(f"  Score Range:    [{min(all_task_scores):.4f}, {max(all_task_scores):.4f}]")

    print(format_header("Simulation Complete", 70))


if __name__ == "__main__":
    main()
