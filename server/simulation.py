"""Bioinformatics simulation tools for CRISPR v2.

All functions are deterministic given a seeded RNG. No external deps beyond numpy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .models import (
    CorrectionResult,
    GuideDesign,
    GuideEvaluation,
    MutationInfo,
    OffTargetHit,
    PAMSite,
)

BASES = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ── Gene generation ──────────────────────────────────────────────────────

def generate_gene(length: int, rng: np.random.Generator, gc_target: float = 0.50) -> str:
    """Generate a DNA sequence with controlled GC content."""
    gc_count = int(length * gc_target)
    at_count = length - gc_count
    bases = (["G"] * (gc_count // 2) + ["C"] * (gc_count - gc_count // 2) +
             ["A"] * (at_count // 2) + ["T"] * (at_count - at_count // 2))
    rng.shuffle(bases)
    return "".join(bases)


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, b) for b in reversed(seq))


def inject_mutations(
    gene: str, positions: List[int], rng: np.random.Generator
) -> Tuple[str, List[MutationInfo]]:
    """Place mutations at specified positions. Returns mutated gene + mutation info."""
    gene_list = list(gene)
    mutations = []
    for pos in positions:
        ref = gene_list[pos]
        alts = [b for b in BASES if b != ref]
        alt = str(rng.choice(alts))
        gene_list[pos] = alt
        mutations.append(MutationInfo(position=pos, ref_base=ref, alt_base=alt))
    return "".join(gene_list), mutations


def generate_genome_context(
    gene: str,
    flanking_length: int,
    rng: np.random.Generator,
    num_off_target_seeds: int = 0,
    guide_seeds: List[str] | None = None,
) -> Tuple[str, List[Dict]]:
    """Generate flanking DNA with planted off-target homology regions.

    Returns (full_context, off_target_site_info_list).
    The full context is: flanking_left + gene + flanking_right.
    """
    left = generate_gene(flanking_length, rng, gc_target=0.48)
    right = generate_gene(flanking_length, rng, gc_target=0.48)
    context = left + gene + right
    gene_offset = flanking_length  # gene starts at this index in context

    off_target_sites: List[Dict] = []

    # Plant off-target sites that resemble guide sequences (with mismatches)
    if guide_seeds and num_off_target_seeds > 0:
        context_list = list(context)
        for _ in range(num_off_target_seeds):
            seed_guide = str(rng.choice(guide_seeds))
            # Pick a random position in the flanking regions (not the gene itself)
            side = rng.choice(["left", "right"])
            if side == "left":
                pos = int(rng.integers(10, flanking_length - 25))
            else:
                pos = int(rng.integers(gene_offset + len(gene) + 10, len(context) - 25))

            # Place the guide with 1-3 mismatches
            num_mismatches = int(rng.integers(1, 4))
            planted = list(seed_guide)
            mismatch_positions = rng.choice(len(planted), size=min(num_mismatches, len(planted)), replace=False)
            for mp in mismatch_positions:
                alts = [b for b in BASES if b != planted[mp]]
                planted[mp] = str(rng.choice(alts))

            for i, b in enumerate(planted):
                if pos + i < len(context_list):
                    context_list[pos + i] = b

            off_target_sites.append({
                "position": pos,
                "mismatches": num_mismatches,
                "original_guide": seed_guide,
            })

        context = "".join(context_list)

    return context, off_target_sites


# ── PAM site search ──────────────────────────────────────────────────────

def search_pam_sites(gene_sequence: str, pam_pattern: str) -> List[PAMSite]:
    """Search for PAM motifs in the gene. Supports NGG, NNGRRT patterns."""
    sites = []
    pam_len = len(pam_pattern)

    def matches_pattern(subseq: str, pattern: str) -> bool:
        for s, p in zip(subseq, pattern):
            if p == "N":
                continue
            elif p == "R":
                if s not in ("A", "G"):
                    return False
            elif s != p:
                return False
        return True

    # Forward strand: PAM is downstream of guide (3' end)
    for i in range(len(gene_sequence) - pam_len + 1):
        subseq = gene_sequence[i : i + pam_len]
        if matches_pattern(subseq, pam_pattern):
            # Guide would be 20nt upstream of PAM position
            if i >= 20:
                sites.append(PAMSite(
                    position=i,
                    strand="+",
                    pattern=pam_pattern,
                    sequence=subseq,
                ))

    # Reverse strand: check reverse complement
    rc = reverse_complement(gene_sequence)
    for i in range(len(rc) - pam_len + 1):
        subseq = rc[i : i + pam_len]
        if matches_pattern(subseq, pam_pattern):
            orig_pos = len(gene_sequence) - i - pam_len
            if orig_pos + pam_len + 20 <= len(gene_sequence):
                sites.append(PAMSite(
                    position=orig_pos,
                    strand="-",
                    pattern=pam_pattern,
                    sequence=subseq,
                ))

    return sites


# ── Guide design ─────────────────────────────────────────────────────────

def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in ("G", "C"))
    return gc / len(seq)


def _distance_to_nearest_mutation(position: int, mutations: List[MutationInfo]) -> int:
    if not mutations:
        return 9999
    return min(abs(position - m.position) for m in mutations)


def design_guide_at_pam(
    gene_sequence: str,
    pam_position: int,
    strand: str,
    mutations: List[MutationInfo],
    guide_counter: int,
) -> GuideDesign | None:
    """Design a 20nt guide RNA targeting a PAM site."""
    if strand == "+":
        start = pam_position - 20
        end = pam_position
        if start < 0:
            return None
        guide_seq = gene_sequence[start:end]
    else:
        start = pam_position + len("NGG")  # after PAM
        end = start + 20
        if end > len(gene_sequence):
            return None
        guide_seq = reverse_complement(gene_sequence[start:end])

    gc = _gc_content(guide_seq)
    dist = _distance_to_nearest_mutation(pam_position, mutations)

    # On-target score: based on distance to mutation + GC quality
    distance_score = max(0.0, 1.0 - dist / 50.0)  # drops off over 50bp
    gc_score = 1.0 - min(abs(gc - 0.55) / 0.35, 1.0)  # optimal around 55%
    on_target = 0.6 * distance_score + 0.4 * gc_score

    return GuideDesign(
        guide_id=f"guide_{guide_counter}",
        sequence=guide_seq,
        pam_position=pam_position,
        strand=strand,
        distance_to_nearest_mutation=dist,
        gc_content=round(gc, 3),
        on_target_score=round(on_target, 3),
    )


# ── Guide evaluation ─────────────────────────────────────────────────────

def _max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _self_complementarity(seq: str) -> float:
    """Simple palindrome check: fraction of positions that match reverse complement."""
    rc = reverse_complement(seq)
    matches = sum(1 for a, b in zip(seq, rc) if a == b)
    return round(matches / max(len(seq), 1), 3)


def evaluate_guide(guide_sequence: str) -> GuideEvaluation:
    """Detailed quality evaluation of a guide sequence."""
    gc = _gc_content(guide_sequence)
    homo = _max_homopolymer(guide_sequence)
    self_comp = _self_complementarity(guide_sequence)

    # GC quality
    if 0.40 <= gc <= 0.65:
        gc_qual = "optimal"
    elif 0.30 <= gc <= 0.75:
        gc_qual = "suboptimal"
    else:
        gc_qual = "poor"

    # Overall quality
    score = 0
    if gc_qual == "optimal":
        score += 2
    elif gc_qual == "suboptimal":
        score += 1
    if homo <= 3:
        score += 2
    elif homo <= 4:
        score += 1
    if self_comp < 0.4:
        score += 2
    elif self_comp < 0.6:
        score += 1

    if score >= 5:
        quality = "high"
    elif score >= 3:
        quality = "medium"
    else:
        quality = "low"

    return GuideEvaluation(
        guide_id="",
        sequence=guide_sequence,
        gc_content=round(gc, 3),
        gc_quality=gc_qual,
        homopolymer_max_run=homo,
        self_complementarity=self_comp,
        overall_quality=quality,
    )


# ── Off-target scanning ──────────────────────────────────────────────────

def scan_off_targets(
    guide_sequence: str,
    genome_context: str,
    gene_offset: int,
    gene_length: int,
    regulatory_regions: List[Tuple[int, int]],
) -> List[OffTargetHit]:
    """Scan genome context for sequences similar to the guide (<=3 mismatches)."""
    guide_len = len(guide_sequence)
    hits = []

    for i in range(len(genome_context) - guide_len + 1):
        # Skip the on-target region (within the gene near the edit site)
        window = genome_context[i : i + guide_len]
        mismatches = sum(1 for a, b in zip(guide_sequence, window) if a != b)

        if 1 <= mismatches <= 3:
            # Check if this position falls in a regulatory region
            # Map context position back to gene-relative position
            gene_relative = i - gene_offset
            in_reg = False
            for reg_start, reg_end in regulatory_regions:
                if reg_start <= gene_relative <= reg_end:
                    in_reg = True
                    break

            if mismatches == 1:
                risk = "high"
            elif mismatches == 2:
                risk = "medium"
            else:
                risk = "low"

            hits.append(OffTargetHit(
                position=i,
                mismatches=mismatches,
                in_regulatory_region=in_reg,
                risk_level=risk,
            ))

    # Limit to top 20 hits by risk (to keep output manageable)
    risk_order = {"high": 0, "medium": 1, "low": 2}
    hits.sort(key=lambda h: (risk_order[h.risk_level], h.mismatches))
    return hits[:20]


# ── Edit simulation ──────────────────────────────────────────────────────

def simulate_edit(
    guide_sequence: str,
    target_position: int,
    mutations: List[MutationInfo],
    off_target_hits: List[OffTargetHit],
    on_target_score: float,
    rng: np.random.Generator,
) -> Tuple[List[CorrectionResult], List[OffTargetHit]]:
    """Simulate applying a CRISPR edit.

    Returns (corrections, off_target_damage).
    """
    corrections = []
    guide_len = len(guide_sequence)

    # On-target: try to correct mutations within ~30bp of target
    for mut in mutations:
        dist = abs(mut.position - target_position)
        if dist <= 30:
            # Success probability based on on-target score and distance
            p_success = on_target_score * max(0.0, 1.0 - dist / 40.0)
            corrected = float(rng.random()) < p_success
            corrections.append(CorrectionResult(
                mutation_position=mut.position,
                corrected=corrected,
            ))

    # Off-target damage
    damage = []
    for hit in off_target_hits:
        # Probability of off-target cleavage decreases with mismatches
        if hit.mismatches == 1:
            p_damage = 0.85
        elif hit.mismatches == 2:
            p_damage = 0.35
        else:
            p_damage = 0.10

        if float(rng.random()) < p_damage:
            damage.append(hit)

    return corrections, damage


# ── Sequence analysis ────────────────────────────────────────────────────

def analyze_sequence_region(gene: str, start: int, end: int) -> Dict:
    """Analyze a region of the gene for GC content, repeats, and structure hints."""
    region = gene[start:end]
    gc = _gc_content(region)
    homo = _max_homopolymer(region)

    # Dinucleotide repeats
    dinuc_repeats = 0
    for i in range(len(region) - 3):
        if region[i:i+2] == region[i+2:i+4]:
            dinuc_repeats += 1

    # Simple secondary structure proxy: self-complementarity of the region
    self_comp = _self_complementarity(region)

    return {
        "region": f"{start}-{end}",
        "length": end - start,
        "sequence": region if len(region) <= 60 else region[:30] + "..." + region[-30:],
        "gc_content": round(gc, 3),
        "max_homopolymer_run": homo,
        "dinucleotide_repeats": dinuc_repeats,
        "self_complementarity": round(self_comp, 3),
        "structure_risk": "high" if self_comp > 0.5 else "moderate" if self_comp > 0.35 else "low",
    }
