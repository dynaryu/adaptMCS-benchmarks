from rips.utils import *
from typing import List

__all__ = ['gs']


def get_root_particle(fk: FeynmanKac, cloud: Cloud) -> Particle:
    x0 = fk.sample_particle()
    if cloud is not None:
        cloud.particles.append(x0)
    return x0


def gs(fk: FeynmanKac, kernels: Kernels, cloud: Cloud, lineage: bool = False) -> float:
    """Generalized splitting for Feynman-Kac models.

    Args:
        fk: Feynman-Kac model.
        kernels:
        cloud: sample particles

    Returns:
        Unbiased Monte Carlo unreliability sample.
    """
    # Root particle
    x0 = get_root_particle(fk, cloud)
    X = [x0]

    # Evolve particles
    for level in range(1, fk.levels):
        X0 = [x for x in X if x.score >= fk.scores[level]]
        score = fk.scores[level]
        X = [x for x in resample(fk, kernels, X0, level, score, lineage=lineage)]

    pgs = sum(x.score >= 1 for x in X) * fk.s_factor ** - (fk.levels - 1)

    return pgs

