from typing import Dict

import numpy as np

from rips.utils import *
from rips.kernels.metropolis import Metropolis

__all__ = ['Adaptive']


class Adaptive(Metropolis):
    """Supports adaptive kernels with a single scaling parameter."""

    def __init__(
            self,
            target_ar: float = .234,
            chain_length: int = 100,
            scaling0: float = 0.3,
            **kv):
        self.adaptive: Dict[Level, bool] = {0: True}
        """Signals what levels are adaptive."""
        self.target_ar: float = target_ar
        """Target acceptance rate."""
        self.chain_length: int = chain_length
        """Number of transitions before updating adaptive parameters."""
        self.scaling0: float = scaling0
        """Initial scaling."""
        self.scaling: Dict[Level, float] = {0: self.scaling0}
        """Scaling parameter for each level."""
        self.chains: Dict[Level, int] = {0: 0}
        """Number of completed chains."""
        super().__init__(**kv)
        self.actual_chain_length = self.burn * self.chain_length

    def add_level(self, level: Level):
        super().add_level(level)
        self.scaling[level] = self.scaling[level - 1]
        self.adaptive[level] = self.adaptive[level - 1]
        self.adaptive[level - 1] = False
        self.chains[level] = 0

    def iterations(self, level: Level) -> int:
        """Returns number of kernel iterations for current level."""
        return self.accepts[level] + self.rejects[level]

    def sample_acceptance_rate(self, level: Level) -> float:
        """Returns sample acceptance rate."""
        return self.accepts[level] / max(1, self.iterations(level))

    def update_scaling(self, level: Level):
        """Updates scaling parameters for a given level."""
        if self.iterations(level) % self.actual_chain_length == 0:
            self.chains[level] += self.iterations(level) / self.actual_chain_length
            log_l = np.log(self.scaling[level])
            log_l += 1 / np.sqrt(self.chains[level]) * (self.sample_acceptance_rate(level) - self.target_ar)
            # print(self.sample_acceptance_rate(level))
            self.scaling[level] = np.exp(log_l)
            self.accepts[level] = 0
            self.rejects[level] = 0

    def sample_transition(
            self,
            fk: FeynmanKac,
            particle: Particle,
            level: Level,
            score: Score
    ) -> Particle:
        particle = super().sample_transition(fk, particle, level, score)
        if self.adaptive[level]:
            self.update_scaling(level)
        return particle

    def stats_level(self, level: Level):
        scaling = self.scaling[level]
        print(f'level={level}, iterations={self.iterations(level)},'
              f' accept_rate={self.sample_acceptance_rate(level):.2f}, scaling={scaling:.4e}')
