from dataclasses import dataclass
from rips.utils import *
from rips.kernels.adaptive import Adaptive
from rips.kernels.metropolis import *

import numpy as np

__all__ = ['Gaussian', 'NNGaussian']


class Gaussian(Adaptive, Metropolis):
    """Possibly adaptive Metropolis with Gaussian proposal."""

    @property
    def kernel_name(self):
        return 'gauss'

    def proposal_sample(self, fk: FeynmanKac, path: Path, level: Level) -> Path:
        path_new = path + self.scaling[level] * np.random.randn(fk.num_variables)
        return path_new


@dataclass
class NNGaussian(Gaussian):
    """Gaussian proposal for non-negative target distributions."""

    kernel_name: str = 'nng'

    def proposal_sample(self, path: Path, level: Level) -> Path:
        return np.abs(super().proposal_sample(path, level))
