from dataclasses import dataclass

import numpy as np

from rips.utils import *
from rips.kernels.adaptive import Adaptive
from rips.kernels.metropolis import MetropolisHastings

__all__ = ['PCN', 'GaussPCN', 'NNPCN']


class PCN(Adaptive, MetropolisHastings):
    """Preconditioned Crank-Nicholson"""

    @property
    def kernel_name(self) -> str:
        return 'pcn'

    def proposal_logpdf(self, fk: FeynmanKac, path_new: Path, path: Path,
                        level: int) -> float:
        sigma = self.scaling[level]
        mu = path * np.sqrt(1 - sigma ** 2)
        return (np.log(1 / sigma / np.sqrt(2 * np.pi)) - 1 / 2 * (
                    (path_new - mu) / sigma) ** 2).sum()

    def proposal_sample(self, fk: FeynmanKac, path: Path, level: Level) -> Path:
        sigma = self.scaling[level]
        mu = path * np.sqrt(1 - sigma ** 2)
        path_new = mu + sigma * np.random.standard_normal(fk.num_variables)
        return path_new

    def update_scaling(self, level: Level):
        super().update_scaling(level)
        self.scaling[level] = min(1.0, self.scaling[level])
        return

    def acceptance_rate(self, fk: FeynmanKac, path: Path, path_new: Path,
                        level: int = 0) -> float:
        if fk.measure_name == 'std':
            return 1.0
        return super().acceptance_rate(fk, path, path_new, level)


class GaussPCN(PCN):
    def acceptance_rate(self, fk: FeynmanKac, path: Path, path_new: Path,
                        level: int = 0) -> float:
        return 1.0


@dataclass
class NNPCN(PCN):
    """PCN for non-negative random variables."""

    kernel_name: str = 'nnpcn'

    def proposal_sample(self, fk: FeynmanKac, path: Path, level: Level) -> Path:
        return np.abs(super().proposal_sample(fk, path, level))
