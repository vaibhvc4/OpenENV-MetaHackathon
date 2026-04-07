from __future__ import annotations

from statistics import mean

from agents.baseline import BaselineAgent
from env.environment import CrisprEnv


def run_task(task_name: str, seed: int, episodes: int = 5) -> tuple[float, float]:
    rewards = []
    scores = []
    agent = BaselineAgent()

    for episode in range(episodes):
        env = CrisprEnv(task_level=task_name, seed=seed + episode)
        total_reward, info, _trace = agent.run_episode(env)
        rewards.append(total_reward)
        scores.append(float(info.get("final_score", 0.0)))

    return mean(rewards), mean(scores)


def main() -> None:
    seed = 123
    tasks = ["easy", "medium", "hard"]

    print(f"Running baseline with fixed seed={seed}")
    print("-" * 56)

    avg_rewards = []
    avg_scores = []

    for task in tasks:
        avg_reward, avg_score = run_task(task_name=task, seed=seed, episodes=5)
        avg_rewards.append(avg_reward)
        avg_scores.append(avg_score)
        print(f"Task={task:>6} | avg_reward={avg_reward:.4f} | avg_score={avg_score:.4f}")

    print("-" * 56)
    print(f"Overall avg reward: {mean(avg_rewards):.4f}")
    print(f"Overall avg score : {mean(avg_scores):.4f}")


if __name__ == "__main__":
    main()
