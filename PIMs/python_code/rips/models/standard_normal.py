from rips.utils import *
import numpy as np

__all__ = ['StandardNormal']


class StandardNormal(Model):
    """Multivariate standard gaussian random variables."""

    @property
    def measure_name(self):
        return 'std'

    def logpdf(self, path: Path) -> float:
        return np.log(1 / np.sqrt(2 * np.pi)) * self.num_variables\
               - 1 / 2 * (path ** 2).sum()

    def logmarginals(self, path: Path) -> np.ndarray:
        return np.log(1 / np.sqrt(2 * np.pi)) - 1 / 2 * path ** 2

    def sample(self) -> Path:
        return np.random.standard_normal(self.num_variables)

    def sample_marginal(self) -> float:
        return np.random.standard_normal()
