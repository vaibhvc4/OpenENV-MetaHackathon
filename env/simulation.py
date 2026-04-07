from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .models import GuideRNA, Mutation

DNA_BASES = np.array(["A", "C", "G", "T"])


def random_dna_sequence(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(DNA_BASES, size=length).tolist())


def inject_substitution_mutations(
    sequence: str,
    mutation_positions: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[str, List[Mutation]]:
    seq_list = list(sequence)
    mutations: List[Mutation] = []

    for position in mutation_positions:
        original = seq_list[position]
        alternatives = [b for b in DNA_BASES.tolist() if b != original]
        mutated = str(rng.choice(alternatives))
        seq_list[position] = mutated
        mutations.append(
            Mutation(
                position=position,
                original_base=original,
                mutated_base=mutated,
            )
        )

    return "".join(seq_list), mutations


def _gc_content(sequence: str) -> float:
    gc_count = sum(1 for base in sequence if base in {"G", "C"})
    return gc_count / max(len(sequence), 1)


def _calc_efficiency(match_ratio: float, guide_seq: str, rng: np.random.Generator) -> float:
    gc = _gc_content(guide_seq)
    gc_bonus = 1.0 - min(abs(gc - 0.5) / 0.5, 1.0)
    noise = float(rng.uniform(-0.03, 0.03))
    efficiency = 0.2 + 0.6 * match_ratio + 0.2 * gc_bonus + noise
    return float(np.clip(efficiency, 0.0, 1.0))


def _calc_off_target_risk(guide_seq: str, match_ratio: float, rng: np.random.Generator) -> float:
    repeats = sum(1 for i in range(len(guide_seq) - 1) if guide_seq[i] == guide_seq[i + 1])
    repeat_penalty = repeats / max(len(guide_seq) - 1, 1)
    noise = float(rng.uniform(-0.03, 0.03))
    risk = 0.15 + 0.35 * (1.0 - match_ratio) + 0.45 * repeat_penalty + noise
    return float(np.clip(risk, 0.0, 1.0))


def generate_candidate_guides(
    mutated_sequence: str,
    mutations: Sequence[Mutation],
    rng: np.random.Generator,
    guide_length: int = 20,
    guides_per_mutation: int = 3,
    scenario_bias: str = "balanced",
) -> List[GuideRNA]:
    guides: List[GuideRNA] = []
    seq_len = len(mutated_sequence)
    guide_index = 0

    for mutation in mutations:
        starts: List[int] = []
        center_start = max(0, mutation.position - guide_length // 2)
        center_start = min(center_start, seq_len - guide_length)
        starts.append(center_start)

        while len(starts) < guides_per_mutation:
            jitter = int(rng.integers(-6, 7))
            start = int(np.clip(center_start + jitter, 0, seq_len - guide_length))
            if start not in starts:
                starts.append(start)

        for start in starts:
            end = start + guide_length
            guide_seq = mutated_sequence[start:end]
            target_positions = [m.position for m in mutations if start <= m.position < end]
            match_ratio = len(target_positions) / max(len(mutations), 1)

            efficiency = _calc_efficiency(match_ratio, guide_seq, rng)
            off_target = _calc_off_target_risk(guide_seq, match_ratio, rng)

            if scenario_bias == "easy" and mutation.position in target_positions and start == center_start:
                efficiency = min(1.0, efficiency + 0.25)
                off_target = max(0.0, off_target - 0.2)
            elif scenario_bias == "hard":
                efficiency = max(0.0, efficiency - float(rng.uniform(0.0, 0.12)))
                off_target = min(1.0, off_target + float(rng.uniform(0.05, 0.18)))

            guide = GuideRNA(
                id=f"g{guide_index}",
                sequence=guide_seq,
                start=start,
                end=end,
                target_positions=target_positions,
                efficiency=float(np.clip(efficiency, 0.0, 1.0)),
                off_target_risk=float(np.clip(off_target, 0.0, 1.0)),
            )
            guides.append(guide)
            guide_index += 1

    dedup: Dict[Tuple[int, int], GuideRNA] = {}
    for guide in guides:
        key = (guide.start, guide.end)
        current = dedup.get(key)
        if current is None or (guide.efficiency - guide.off_target_risk) > (
            current.efficiency - current.off_target_risk
        ):
            dedup[key] = guide

    return list(dedup.values())


def success_probability(efficiency: float, off_target_risk: float) -> float:
    return float(np.clip(efficiency - off_target_risk, 0.0, 1.0))


def compute_corrected_mutations(
    selected_guide: GuideRNA,
    mutations: Sequence[Mutation],
    rng: np.random.Generator,
) -> Iterable[int]:
    p_success = success_probability(selected_guide.efficiency, selected_guide.off_target_risk)
    corrected: List[int] = []

    for mutation in mutations:
        if mutation.position not in selected_guide.target_positions:
            continue
        if float(rng.random()) < p_success:
            corrected.append(mutation.position)

    return corrected
