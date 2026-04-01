import abc
from typing import Dict, List, Iterator

import numpy as np

from rips.utils.base import FKComponent, FeynmanKac
from rips.utils.particle import *
from rips.utils.model import Model


__all__ = [
    'Kernel', 'Kernels', 'resample',
]


class Kernel(FKComponent):
    """Parent class for Markov transition kernels."""

    def __init__(self, burn: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.burn: int = burn
        """Number of Kernel iterations to reduce correlation between samples."""
        self.accepts: Dict[Level, int] = {}
        """Number of kernel samples accepted at each level."""
        self.rejects: Dict[Level, int] = {}
        """Number of kernel samples rejected at each level."""
        self.levels: Level = 1
        """Number of kernel indices."""
        # Initialize
        self.accepts = {0: 0}
        self.rejects = {0: 0}

    @property
    def kernel_name(self) -> str:
        """For keeping track of kernels used."""
        return 'nan'

    @property
    def log_name(self):
        return f'{self.kernel_name}{self.burn}'

    def sample_transition(
            self,
            fk: FeynmanKac,
            particle: Particle,
            level: Level,
            score: float
    ) -> Particle:
        """Samples particle's transition."""
        raise NotImplementedError

    def sample_transition_full(
            self,
            m: Model,
            particle: Particle,
            level: Level,
            score: float
    ) -> Particle:
        """Samples a particle transition with burn in."""
        for _ in range(self.burn):
            particle = self.sample_transition(m, particle, level, score)
        return particle

    def reset_levels(self):
        """Restarts kernel."""
        self.accepts = {0: 0}
        self.rejects = {0: 0}
        self.levels = 0

    def add_level(self, level: Level):
        """Extends data structures to additional transition level.
        """
        self.accepts[level] = 0
        self.rejects[level] = 0
        self.levels += 1

    def stats(self):
        """Prints basic kernel information and statistics.
        """
        print(f'burn={self.burn}')
        for level in range(self.levels + 1):
            self.stats_level(level)

    def stats_level(self, level: Level):
        """Prints basic kernel information and statistics for a given level.
        """
        pass


Kernels = List[Kernel]
"""List of kernels."""


def kernel_iterator(fk: FeynmanKac, kernel: Kernel, particle: Particle,
                    level: int, score) -> Particle:
    """Samples a particle transition iterating kernel to reduce correlation."""
    for _ in range(kernel.burn):
        particle = kernel.sample_transition(fk, particle, level, score)
    return particle


def resample(
        fk: FeynmanKac,
        kernels: Kernels,
        particles: List[Particle],
        level: int,
        score,
        shuffle: bool = False,
        lineage: bool = False,
) -> Iterator[Particle]:
    """Resampling step in splitting algorithms."""
    if shuffle:
        np.random.shuffle(particles)
    for particle in particles:
        new_particle = particle.copy()
        for _ in range(fk.s_factor):
            for kernel in kernels:
                new_particle = kernel_iterator(fk, kernel, new_particle, level, score)
            if lineage:
                new_particle.predecessor = particle
                new_particle.k = particle.k + 1
                particle.successors.append(new_particle)
            yield new_particle
