"""
Metropolis
==========

Implements Metropolis and Metropolis-Hasting Kernels.
"""

import numpy as np
from rips.utils import *

__all__ = ['Metropolis', 'MetropolisHastings', 'Gibbs']


class Metropolis(Kernel):
    """Base class for Kernels with symmetric proposals."""

    def proposal_sample(self, fk: FeynmanKac, path: Path, level: Level) -> Path:
        """ Returns sample path from proposal distribution.
        """
        raise NotImplementedError

    def acceptance_rate(self, fk: FeynmanKac, path: Path, path_new: Path, level: int = 0) -> float:
        """Returns rate of proposed new path.
        """
        return np.exp(fk.logpdf(path_new) - fk.logpdf(path))

    def sample_transition(
            self,
            fk: FeynmanKac,
            particle: Particle,
            level: Level,
            score: Score,
    ) -> Particle:
        path_new = self.proposal_sample(fk, particle.path, level)
        a = self.acceptance_rate(fk, particle.path, path_new)
        if np.random.rand() <= a:
            particle_new = fk.compute_particle(path_new, level)
            if particle_new.score >= score:
                self.accepts[level] += 1
                return particle_new
        self.rejects[level] += 1
        return particle.copy()


class MetropolisHastings(Metropolis):
    """Base class for non-symmetric proposals."""

    def proposal_logpdf(self, fk: FeynmanKac, path_new: Path, path: Path, level: int) -> float:
        """ Log PDF of proposal distribution.
        """
        raise NotImplementedError

    def acceptance_rate(self, fk: FeynmanKac, path: Path, path_new: Path, level: int = 0) -> float:
        log_a1 = fk.logpdf(path_new) - fk.logpdf(path)
        log_a2 = self.proposal_logpdf(fk, path, path_new, level) - self.proposal_logpdf(fk, path_new, path, level)
        return np.exp(log_a1 + log_a2)

import random

class Gibbs(Kernel):

    def sample_transition(self, fk: FeynmanKac, particle: Particle, level: Level) -> Particle:
        new_particle = particle.copy()
        indices = list(range(fk.num_variables))
        random.shuffle(indices)
        for i in indices:
            self.sample_component(fk, new_particle, level, i)
        return new_particle

    def sample_component(self, fk: FeynmanKac, particle: Particle, level: Level, i: int):
        raise NotImplementedError
