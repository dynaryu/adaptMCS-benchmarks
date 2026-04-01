import numpy as np

from rips.utils import *

__all__ = ['smc', ]


def add_score(
        fk: FeynmanKac,
        score: Score,
        kernels: Kernels,
        verbosity_level: int = 0
):
    # Update FK model
    fk.add_score(score)
    # Prepare kernels for next level
    for kernel in kernels:
        kernel.add_level(fk.levels)
    if verbosity_level > 0:
        print(fk.scores)


class SMCLogger:

    def __init__(
            self,
            fk:
            FeynmanKac,
            kernels: Kernels,
            cloud: Cloud = None,
            lineage: bool = False,
            verbosity_level: int = 0
    ):
        self.fk = fk
        self.kernels = kernels
        self.cloud = Cloud() if cloud is None else cloud
        self.lineage = lineage
        self.verbosity_level = verbosity_level

    def get_sample_particles(self):
        if not self.cloud.particles:
            X = [self.fk.sample_particle() for _ in self.particle_counter()]
            self.cloud.particles = X
        else:
            X = self.cloud.particles
        return X

    def resample(self, X):
        score = self.fk.scores[self.fk.levels]
        gen = resample(self.fk, self.kernels, X, self.fk.levels, score, shuffle=True,
                       lineage=self.lineage)
        Xnew = [x for x in gen]
        return Xnew

    def particle_counter(self):
        for i in range(self.fk.num_of_particles):
            yield i


def smc_level_logger(
        fk: FeynmanKac,
        kernels: Kernels,
        cloud: Cloud = None,
        verbosity_level: int = 0
):
    """Saves and/or prints progress for every level.

    Args:
        fk:
        kernels:
        cloud:
        verbosity_level:

    """
    # Logging results
    if cloud is not None:
        cloud.to_file()

    # Printing results
    if verbosity_level > 0:
        print(f'Level: {fk.levels}')
        print(f'Score: {fk.scores[-1]}')
        for kernel in kernels:
            kernel.stats_level(fk.levels)


def smc(
        fk: FeynmanKac,
        kernels: Kernels,
        cloud: Cloud = None,
        lineage: bool = False,
        verbosity_level: int = 1,
) -> float:
    """Sequential Monte Carlo algorithm.

    Args:
        fk: Feynman-Kac model
        kernels: list of probability kernels
        cloud: sample particles
        lineage:
        verbosity_level: higher for more info.

    Returns:
        Biased sample mean.
    """

    # Keeping track of results
    smc_logger = SMCLogger(fk, kernels, cloud, lineage, verbosity_level)

    # Reset past levels if any
    if fk.levels > 0:
        fk.reset_scores()

    # Initialization parameters
    n, s = fk.num_of_particles, fk.s_factor
    q = int(n - n / s)  # Number of low-scoring particles

    # Initial level
    X = smc_logger.get_sample_particles()
    X = sorted(X)

    # Add score
    add_score(fk, (X[q - 1].score + X[q].score) / 2, kernels, verbosity_level)
    # smc_level_logger(fk, kernels, cloud, verbosity_level)

    # Evolve Markov Chain until score reaches 1
    while fk.scores[-1] < 1:
        # Split high-scoring Trajectories
        X = smc_logger.resample(X[q:])
        X = sorted(X)
        # Next score
        add_score(fk, (X[q - 1].score + X[q].score) / 2, kernels,
                  verbosity_level)
    else:
        fk.scores[-1] = 1
        if verbosity_level > 0:
            print(fk.scores)

    p = sum(x.score >= 1 for x in X) / n * s ** - (fk.levels - 1)

    return p
