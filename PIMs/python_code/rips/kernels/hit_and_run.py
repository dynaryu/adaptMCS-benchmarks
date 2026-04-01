from dataclasses import dataclass
from rips.utils import *
from rips.kernels.metropolis import Metropolis
from rips.kernels.adaptive import Adaptive
import numpy as np

__all__ = ['HitAndRun', 'NNHitAndRun']

Direction = np.ndarray


@dataclass
class HitAndRun(Adaptive):
    """Hit-and-run Kernel. Uses Metropolis and a (multivariate) Gaussian distribution for distance (direction)."""
    kernel_name: str = "har"

    def get_direction(self, level: Level) -> Direction:
        """Returns relative direction of proposal sample.

        Args:
            level: transition level.

        """
        z = np.random.randn(self.fk.num_variables)
        z /= np.linalg.norm(z)
        return z

    def get_distance(self, level: Level) -> float:
        """Returns relative distance of proposal sample.

        Args:
            level: transition level.

        """
        return np.random.randn() * self.scaling[level]

    def proposal_sample(self, path: Path, level: Level) -> Path:
        d = self.get_direction(level)
        l = self.get_distance(level)
        path_new = path + l * d
        return path_new


class NNHitAndRun(HitAndRun):

    def proposal_sample(self, path: Path, level: Level) -> Path:
        return np.abs(super().proposal_sample(path, level))
