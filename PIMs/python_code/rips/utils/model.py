import numpy as np
from rips.utils.particle import *
from rips.utils import FKComponent


__all__ = [
    'Model',
]


class Model:
    """Stochastic model."""

    @property
    def num_variables(self) -> int:
        """Number of random variables."""
        raise NotImplementedError

    def sample(self) -> Path:
        """Simulates random process."""
        raise NotImplementedError

    def logpdf(self, path: Path) -> float:
        """Log probability distribution function."""
        raise NotImplementedError

    def logmarginals(self, path: Path) -> np.ndarray:
        """Log marginal distribution function."""
        raise NotImplementedError

    def response(self, path: Path, level: Level) -> Response:
        """Computes response from path simulation.

        Args:
            path: process simulation.
            level:

        """
        raise NotImplementedError

    def score_function(self, particle: Particle) -> float:
        """Returns particle score."""
        raise NotImplementedError

    def scan(self, random: bool = True) -> np.ndarray:
        """Returns variable indices for systematic samplers."""
        if random:
            return np.random.rand(self.num_variables).argsort()
        else:
            return np.arange(self.num_variables)

    @property
    def measure_name(self) -> str:
        """For special cases."""
        return 'nan'
