"""
Base classes for Feynman-Kac models and Kernels.
"""
from __future__ import annotations

import abc
from typing import Dict
from rips.utils.particle import *

import numpy as np

__all__ = [
    'FeynmanKac', 'FKComponent',
]


class FKComponent:
    """To handle multiple inheritance."""

    def __init__(self, **kwargs):
        pass

    def __post_init__(self):
        pass


class FeynmanKac(FKComponent):
    """Base class for Feynman-Kac models."""

    def __init__(
            self,
            s_factor: int = 5,
            num_of_particles: int = 1000,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.s_factor = s_factor
        """Number of Kernel samples to mitigate auto-correlation."""
        self.num_of_particles = num_of_particles
        """Number of particles."""
        self.scores = list()
        """Thresholds for conditional probabilities."""
        self.levels: int = 0
        """Number of conditional probabilities."""
        self.response_calls: Dict[Level, int] = {0: 0}
        """Number of calls to response function."""
        self.__post_init__()

    @property
    def measure_name(self) -> str:
        """For special cases."""
        return 'nan'

    def __post_init__(self):
        super().__post_init__()
        self.scores.append(-np.inf)

    @property
    @abc.abstractmethod
    def num_variables(self) -> int:
        """Number of random variables."""
        pass

    # def sample(self) -> Path:
    #     """Samples random process."""
    #     raise NotImplementedError
    #
    # def logpdf(self, path: Path) -> float:
    #     raise NotImplementedError
    #
    # def logmarginals(self, path: Path) -> np.ndarray:
    #     raise NotImplementedError

    def compute_particle(self, path: Path, level: Level) -> Particle:
        """Returns Particle from path simulation."""
        response = self.response(path, level)
        self.response_calls[level] += 1
        particle = Particle(path=path, response=response)
        score = self.score_function(particle)
        # score = self.score_function(response)
        particle.score = score
        return particle

    def sample_particle(self) -> Particle:
        """Samples from the initial distribution of particles."""
        path = self.sample()
        return self.compute_particle(path, 0)

    @abc.abstractmethod
    def response(self, path: Path, level: Level) -> Response:
        """Computes response from path simulation.

        Args:
            path: process simulation.
            level:

        """
        pass

    @abc.abstractmethod
    def score_function(self, particle: Particle) -> float:
        # def score_function(self, response: Response) -> float:
        """Returns particle score."""
        pass

    def reset_scores(self):
        """Resets scores and levels."""
        self.scores = [-np.inf]
        self.levels = 0

    def add_score(self, score: Score):
        """Adds score/level to the system of particles."""
        self.scores.append(score)
        # print(self.scores)
        self.levels += 1
        self.response_calls[self.levels] = 0

    def scan(self, random: bool = True) -> np.ndarray:
        """Returns variable indices for systematic samplers."""
        if random:
            return np.random.rand(self.num_variables).argsort()
        else:
            return np.arange(self.num_variables)

    @property
    def total_response_calls(self):
        return sum(value for value in self.response_calls.values())
