from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .models import GeneSequence, GuideRNA, Mutation
from .simulation import generate_candidate_guides, inject_substitution_mutations, random_dna_sequence


@dataclass
class TaskScenario:
    name: str
    reference_gene: GeneSequence
    mutated_gene: GeneSequence
    mutations: List[Mutation]
    candidate_guides: List[GuideRNA]
    max_steps: int
    noisy_observation: bool = False


TaskGenerator = Callable[[np.random.Generator], TaskScenario]
TaskGrader = Callable[[int, int, float], float]


def easy_generator(rng: np.random.Generator) -> TaskScenario:
    reference = random_dna_sequence(length=120, rng=rng)
    mutation_positions = [int(rng.integers(30, 90))]
    mutated, mutations = inject_substitution_mutations(reference, mutation_positions, rng)
    guides = generate_candidate_guides(
        mutated_sequence=mutated,
        mutations=mutations,
        rng=rng,
        guides_per_mutation=4,
        scenario_bias="easy",
    )

    return TaskScenario(
        name="easy",
        reference_gene=GeneSequence(id="easy_ref", sequence=reference),
        mutated_gene=GeneSequence(id="easy_mut", sequence=mutated),
        mutations=mutations,
        candidate_guides=guides,
        max_steps=6,
        noisy_observation=False,
    )


def medium_generator(rng: np.random.Generator) -> TaskScenario:
    reference = random_dna_sequence(length=150, rng=rng)
    mutation_positions = [int(rng.integers(35, 115))]
    mutated, mutations = inject_substitution_mutations(reference, mutation_positions, rng)
    guides = generate_candidate_guides(
        mutated_sequence=mutated,
        mutations=mutations,
        rng=rng,
        guides_per_mutation=6,
        scenario_bias="balanced",
    )

    return TaskScenario(
        name="medium",
        reference_gene=GeneSequence(id="medium_ref", sequence=reference),
        mutated_gene=GeneSequence(id="medium_mut", sequence=mutated),
        mutations=mutations,
        candidate_guides=guides,
        max_steps=8,
        noisy_observation=False,
    )


def hard_generator(rng: np.random.Generator) -> TaskScenario:
    reference = random_dna_sequence(length=180, rng=rng)
    mutation_positions = sorted(rng.choice(np.arange(25, 155), size=3, replace=False).tolist())
    mutated, mutations = inject_substitution_mutations(reference, mutation_positions, rng)
    guides = generate_candidate_guides(
        mutated_sequence=mutated,
        mutations=mutations,
        rng=rng,
        guides_per_mutation=5,
        scenario_bias="hard",
    )

    return TaskScenario(
        name="hard",
        reference_gene=GeneSequence(id="hard_ref", sequence=reference),
        mutated_gene=GeneSequence(id="hard_mut", sequence=mutated),
        mutations=mutations,
        candidate_guides=guides,
        max_steps=10,
        noisy_observation=True,
    )


def easy_grader(corrected: int, total: int, cumulative_reward: float) -> float:
    accuracy = corrected / max(total, 1)
    return float(np.clip(0.8 * accuracy + 0.2 * cumulative_reward, 0.0, 1.0))


def medium_grader(corrected: int, total: int, cumulative_reward: float) -> float:
    accuracy = corrected / max(total, 1)
    return float(np.clip(0.7 * accuracy + 0.3 * cumulative_reward, 0.0, 1.0))


def hard_grader(corrected: int, total: int, cumulative_reward: float) -> float:
    accuracy = corrected / max(total, 1)
    return float(np.clip(0.6 * accuracy + 0.4 * cumulative_reward, 0.0, 1.0))


TASK_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "easy": {"generator": easy_generator, "grader": easy_grader},
    "medium": {"generator": medium_generator, "grader": medium_grader},
    "hard": {"generator": hard_generator, "grader": hard_grader},
}
