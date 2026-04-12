"""Task generators for 3 qualitatively different CRISPR scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .models import MutationInfo, PAMSite
from .simulation import (
    generate_gene,
    generate_genome_context,
    inject_mutations,
    search_pam_sites,
)


@dataclass
class TaskScenario:
    name: str
    task_type: str
    description: str
    gene_sequence: str
    gene_id: str
    mutations: List[MutationInfo]
    genome_context: str
    gene_offset: int  # where the gene starts within genome_context
    regulatory_regions: List[Tuple[int, int]]  # gene-relative positions
    off_target_site_info: List[Dict]
    experiment_budget: int
    max_steps: int


# ── Task 1: Single-Target Knockout (easy) ────────────────────────────────

def single_target_generator(rng: np.random.Generator) -> TaskScenario:
    """One mutation, adequate budget, small genome context. Agent must find PAM,
    design guide, evaluate, and apply."""
    gene = generate_gene(500, rng, gc_target=0.50)
    mut_pos = int(rng.integers(80, 420))
    mutated_gene, mutations = inject_mutations(gene, [mut_pos], rng)

    # Find PAM sites so we can seed off-target homology
    pam_sites = search_pam_sites(mutated_gene, "NGG")
    near_pam = [s for s in pam_sites if abs(s.position - mut_pos) <= 40]

    # Build guide seeds from nearby PAM sites for off-target planting
    guide_seeds = []
    for ps in near_pam[:3]:
        if ps.strand == "+":
            start = ps.position - 20
            if start >= 0:
                guide_seeds.append(mutated_gene[start:ps.position])
        else:
            start = ps.position + 3
            if start + 20 <= len(mutated_gene):
                guide_seeds.append(mutated_gene[start:start + 20])

    context, ot_info = generate_genome_context(
        mutated_gene, flanking_length=1000, rng=rng,
        num_off_target_seeds=3, guide_seeds=guide_seeds or ["A" * 20],
    )

    return TaskScenario(
        name="single_target",
        task_type="single_target",
        description=(
            "A pathogenic single-nucleotide variant has been identified in gene BRCA_SIM. "
            "Your task: correct this mutation using CRISPR-Cas9. Use the available "
            "bioinformatics tools to find PAM sites, design a guide RNA, evaluate its "
            "quality, and apply the edit. Budget is adequate but not unlimited."
        ),
        gene_sequence=mutated_gene,
        gene_id="BRCA_SIM",
        mutations=mutations,
        genome_context=context,
        gene_offset=1000,
        regulatory_regions=[],
        off_target_site_info=ot_info,
        experiment_budget=20,
        max_steps=25,
    )


# ── Task 2: Multi-Mutation Repair (medium) ───────────────────────────────

def multi_repair_generator(rng: np.random.Generator) -> TaskScenario:
    """Three mutations, limited budget. Agent must plan which to group and how
    to allocate resources. Can't fix all 3 individually (3 x 3 credits = 9,
    plus evaluation costs)."""
    gene = generate_gene(1000, rng, gc_target=0.52)

    # Place 3 mutations: two close together (groupable), one far away
    cluster_center = int(rng.integers(200, 400))
    m1 = cluster_center
    m2 = cluster_center + int(rng.integers(15, 35))  # close to m1
    m3 = int(rng.integers(650, 850))  # far from cluster

    mutated_gene, mutations = inject_mutations(gene, [m1, m2, m3], rng)

    pam_sites = search_pam_sites(mutated_gene, "NGG")
    guide_seeds = []
    for ps in pam_sites[:5]:
        if ps.strand == "+":
            start = ps.position - 20
            if start >= 0:
                guide_seeds.append(mutated_gene[start:ps.position])

    context, ot_info = generate_genome_context(
        mutated_gene, flanking_length=2500, rng=rng,
        num_off_target_seeds=6, guide_seeds=guide_seeds or ["A" * 20],
    )

    return TaskScenario(
        name="multi_repair",
        task_type="multi_repair",
        description=(
            "Gene TP53_SIM has three pathogenic mutations that need correction. "
            "Two mutations are in the same exon (close together), and one is in a "
            "different exon. Your experiment budget is limited — you cannot afford "
            "to fix each mutation individually with full evaluation. Plan your edits "
            "strategically: a guide near two close mutations may correct both in one "
            "edit. Prioritize which mutations to fix if budget runs low."
        ),
        gene_sequence=mutated_gene,
        gene_id="TP53_SIM",
        mutations=mutations,
        genome_context=context,
        gene_offset=2500,
        regulatory_regions=[],
        off_target_site_info=ot_info,
        experiment_budget=15,
        max_steps=30,
    )


# ── Task 3: Precision Editing with Constraints (hard) ────────────────────

def precision_editing_generator(rng: np.random.Generator) -> TaskScenario:
    """Two mutations near a regulatory region (no-edit zone). The obvious closest
    PAM site produces a guide that has off-targets in the regulatory region.
    Agent must discover this via off_target_scan and find a safer alternative."""
    gene = generate_gene(1500, rng, gc_target=0.50)

    # Place regulatory region
    reg_start = int(rng.integers(400, 600))
    reg_end = reg_start + int(rng.integers(80, 120))

    # Place mutations near the regulatory region (makes the obvious guide risky)
    m1 = reg_start - int(rng.integers(20, 50))  # just upstream of reg region
    m2 = reg_end + int(rng.integers(20, 50))     # just downstream

    mutated_gene, mutations = inject_mutations(gene, [m1, m2], rng)

    # Find closest PAM sites to each mutation — these are the "trap" PAMs
    pam_sites = search_pam_sites(mutated_gene, "NGG")

    gene_list = list(mutated_gene)
    planted_ot_info = []

    # For each mutation, find the closest PAM sites and plant their guide
    # sequences (with 1 mismatch) inside the regulatory region
    trap_pams = []
    for mut_pos in [m1, m2]:
        nearby = sorted(pam_sites, key=lambda s: abs(s.position - mut_pos))
        for ps in nearby[:3]:  # top 3 closest PAMs per mutation
            if ps.strand == "+" and ps.position >= 20:
                guide_seq = mutated_gene[ps.position - 20 : ps.position]
                trap_pams.append((ps, guide_seq))

    # Plant each trap guide into the regulatory region with exactly 1 mismatch
    reg_len = reg_end - reg_start
    plant_offset = 3
    for _, guide_seq in trap_pams[:4]:
        if reg_len < 25 or plant_offset + 20 >= reg_len:
            break
        pos = reg_start + plant_offset
        planted = list(guide_seq[:20])
        # Single mismatch at a random position
        mm_pos = int(rng.integers(0, len(planted)))
        alts = [b for b in ["A", "C", "G", "T"] if b != planted[mm_pos]]
        planted[mm_pos] = str(rng.choice(alts))
        for idx, b in enumerate(planted):
            if pos + idx < len(gene_list):
                gene_list[pos + idx] = b
        planted_ot_info.append({
            "position": pos, "mismatches": 1, "in_regulatory": True,
        })
        plant_offset += 22  # space out the plantings

    mutated_gene = "".join(gene_list)

    # Also collect safe guides (far from mutations) for genome context seeding
    safe_guides = []
    for ps in pam_sites:
        if abs(ps.position - m1) > 60 and abs(ps.position - m2) > 60:
            if ps.strand == "+" and ps.position >= 20:
                safe_guides.append(mutated_gene[ps.position - 20 : ps.position])

    context, ot_info = generate_genome_context(
        mutated_gene, flanking_length=5000, rng=rng,
        num_off_target_seeds=10,
        guide_seeds=([g for _, g in trap_pams[:3]] + safe_guides[:2]) or ["A" * 20],
    )

    return TaskScenario(
        name="precision_editing",
        task_type="precision_editing",
        description=(
            "Gene CFTR_SIM has two pathogenic mutations flanking a critical splice-site "
            "regulatory region (positions {reg_start}-{reg_end}). This region MUST NOT "
            "be disrupted — any off-target edits there will render the gene non-functional. "
            "The obvious nearby PAM sites produce guides with off-target risk in the "
            "regulatory region. You must find safer alternatives. Use off_target_scan "
            "before applying any edit. Budget is tight."
        ).format(reg_start=reg_start, reg_end=reg_end),
        gene_sequence=mutated_gene,
        gene_id="CFTR_SIM",
        mutations=mutations,
        genome_context=context,
        gene_offset=5000,
        regulatory_regions=[(reg_start, reg_end)],
        off_target_site_info=ot_info + planted_ot_info,
        experiment_budget=12,
        max_steps=35,
    )


# ── Task registry ────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, Dict] = {
    "single_target": {
        "generator": single_target_generator,
        "display_name": "Single-Target Knockout",
        "difficulty": "easy",
    },
    "multi_repair": {
        "generator": multi_repair_generator,
        "display_name": "Multi-Mutation Repair",
        "difficulty": "medium",
    },
    "precision_editing": {
        "generator": precision_editing_generator,
        "display_name": "Precision Editing with Constraints",
        "difficulty": "hard",
    },
}
