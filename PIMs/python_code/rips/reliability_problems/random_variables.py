from typing import Callable, List

import numpy as np
from scipy.stats import multivariate_normal

__all__ = ['PDF','LSF', 'SystemReliability', 'MVN']

import abc


class PDF(metaclass=abc.ABCMeta):
    """
    Joint probability distribution.
    """
    n: int
    """Number of random variables."""

    @abc.abstractmethod
    def sample(self):
        """Returns sample of random variables."""
        pass


class LSF(metaclass=abc.ABCMeta):
    """
    Limit-state function.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class MVN(PDF):
    """Multivariate normal distribution."""

    def __init__(self, mu, sigma=None):
        """
        Args:
            mu: mean or location array of size (n).
            sigma: covariance matrix of size (n, n).
        """
        self.mu = mu
        if sigma is None:
            sigma = np.eye(len(mu))
        self.sigma = sigma
        self.mvn = multivariate_normal

    def sample(self) -> List[float]:
        return multivariate_normal.rvs(self.mu, self.sigma).tolist()


class SystemReliability:
    """Class for system reliability problems with random variables (rv) and a limit state function (lsf)."""

    dual: bool = True
    """If true (false) methods quantify unreliability (reliability)"""

    def __init__(self, rv: PDF, lsf: Callable):
        """
        Args:
            rv: random variables.
            lsf: limit state function.
        """
        self.rv: PDF = rv
        self.lsf: Callable = lsf

    def sample(self):
        return self.rv.sample()

    def phi(self, x):
        return self.lsf(x) > 0



