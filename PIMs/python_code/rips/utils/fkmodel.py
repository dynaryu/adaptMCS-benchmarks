from collections import defaultdict
from typing import Iterator, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np

from rips.utils.kernel import *
from rips.utils.model import *
from rips.utils.particle import *

__all__ = ['FKModel']


class FKModel:

    def __init__(
            self,
            model: Model,
            kernels: Kernels,
            num_particles: int = 10000,
            s: int = 5,
            keep_lineage: bool = False,
    ):
        """Feynman-Kac model for integration.

        Args:
            model: Underlying physical model.
            kernels: Markov transition kernels.
            num_particles: Number of particle trajectories.
            s: Splitting factor
        """
        self.model = model
        self.kernels = kernels
        self.num_particles = num_particles
        self.s = s
        self.scores = [-np.inf, ]
        self.levels = 0
        self.cloud = None
        self.lsf_calls: Dict[Level, int] = defaultdict(lambda: 0)
        """Number of calls to response function."""
        self.keep_lineage: bool = keep_lineage

    def compute_particle(self, path: Path, level: Level) -> Particle:
        """Returns Particle from path simulation."""
        response = self.model.response(path, level)
        self.lsf_calls[level] += 1
        particle = Particle(path=path, response=response)
        particle.score = self.model.score_function(particle)
        particle.keep_lineage = self.keep_lineage
        return particle

    def sample_particle(self) -> Particle:
        """Samples from the initial distribution of particles."""
        path = self.model.sample()
        particle = self.compute_particle(path, 0)
        return particle

    def sample_particles(self, n) -> Particles:
        """Samples n independent particles."""
        particles = [self.sample_particle() for _ in range(n)]
        if self.keep_lineage:
            self.cloud = particles
        return particles

    def sample_transitions(
            self,
            particles: Particles,
            level: int,
            score: float,
            shuffle: bool = False,
    ) -> Iterator[Particle]:
        """Samples particle transitions."""
        if shuffle:
            np.random.shuffle(particles)
            # random.shuffle(particles)
        for particle in particles:
            for _ in range(self.s):
                new_particle = particle.copy()
                for k in self.kernels:
                    new_particle = k.sample_transition_full(
                        self,
                        new_particle,
                        level,
                        score,
                    )
                if self.keep_lineage:
                    new_particle.lineage(particle)
                yield new_particle

    def add_score(self, score: Score):
        """Adds score/level to the system of particles."""
        self.scores.append(score)
        self.levels += 1
        for kernel in self.kernels:
            kernel.add_level(level=self.levels)

    def plot_function(self, f: Callable = None):
        if f is None:
            f = lambda x: x.k + np.random.rand() * .5 - .5
            # f = lambda x: x.score
        x = [part.score for part in self.cloud_leaves]
        y = [f(part) for part in self.cloud_leaves]
        fg, ax = plt.subplots()
        ax.plot(x, y, '.')
        ax.set(xlabel='Importance score')
        plt.show()

    def plot_score_distribution(self):
        scores = np.array([part.score for part in self.cloud_leaves])
        weights = [self.s ** - part.k for part in self.cloud_leaves]
        fg, ax = plt.subplots()
        ax.hist(scores, weights=weights, density=True)
        ax.set(
            xlabel='Importance score',
            # xscale='log',
            yscale='log'
        )
        plt.show()

    @property
    def cloud_leaves(self):
        return (part for root in self.cloud for part in root.leaves())

    def integrate(self, f: Callable):
        s = sum(f(part) * self.s ** - part.k
                for part in self.cloud_leaves)
        return s / len(self.cloud)

    @property
    def num_variables(self):
        return self.model.num_variables
