from rips.utils import *


__all__ = ['annealed_pim']


def annealed_pim_single(fk: FKModel, part: Particle):

    # Initialize particles
    parts = [part]
    level = 1

    # Iterate particles
    for level, score in enumerate(fk.scores[1:-1], start=level):

        # Keep crossing trajectories only
        parts = [part for part in parts if part.score >= score]

        # Split trajectories
        parts = list(fk.sample_transitions(parts, level, score))

    p = sum(part.score >= 1 for part in parts) * fk.s ** - level

    return p


def annealed_pim(fk: FKModel, n: int = None):

    if n is None:
        n = fk.num_particles

    parts = fk.sample_particles(n)

    p = sum(annealed_pim_single(fk, part) for part in parts) / n

    return p
