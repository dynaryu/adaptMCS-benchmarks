from rips.utils.particle import *
from rips.utils.base import *

import numpy as np
import pickle
from typing import Callable


__all__ = ['Cloud', ]


class Cloud:
    """For operations with clouds of particles and other I/O tasks."""

    def __init__(
            self,
            particles: Particles = None,
            file_name: str = None
    ):
        self.particles = [] if particles is None else particles
        self.file_name = file_name
        try:
            if self.file_name is not None:
                self.from_file()
        except FileNotFoundError:
            pass

    def s_score(self, fk: FeynmanKac):
        n, s = fk.num_of_particles, fk.s_factor
        q = int(n - n / s)  # Number of low-scoring particles
        X = sorted(self.particles)
        return (X[q - 1].score + X[q].score) / 2

    def to_file(self):
        """Saves cloud in pickle format."""
        if self.file_name is not None:
            with open(self.file_name, 'wb') as write_file:
                pickle.dump(self.particles, write_file)

    def from_file(self):
        """Loads cloud in pickle format."""
        with open(self.file_name, 'rb') as read_file:
            self.particles = pickle.load(read_file)

    def mean_apply(self, f: Callable):
        return np.mean([p.recursive_apply(f) for p in self.particles]).item()

    @property
    def get_leafs(self):
        def recursive_leafs(particle: Particle):
            if not particle.successors:
                yield particle
            else:
                for successor in particle.successors:
                    yield from recursive_leafs(successor)

        leafs = [leaf for particle in self.particles for leaf in
                 recursive_leafs(particle)]

        return leafs

    @property
    def statisfied_leafs(self):
        leafs = {}

        def seek_recursive_leafs(particle: Particle):
            try:
                leafs[particle.k].append(particle)
            except KeyError:
                leafs[particle.k] = [particle]
            for successor in particle.successors:
                seek_recursive_leafs(successor)

        for p in self.particles:
            seek_recursive_leafs(p)
        return leafs

    def level_particles(self, k):

        def next_particles(particles):
            return [s for p in particles for s in p.successors]

        x = self.particles
        for k in range(k):
            x = next_particles(x)

        return x
