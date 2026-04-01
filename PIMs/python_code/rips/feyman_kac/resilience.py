from typing import Any, Optional, Dict
import abc

import numpy as np

from rips.base import *

__all__ = [
    # Classes
    'Gibbs', 'BinarySystem']


class Gibbs(FeynmanKacConstrained):
    kernel_name = 'gibbs'

    @abc.abstractmethod
    def pdf_marginal_sample(self, variable: int, score) -> float:
        pass

    def update_particle(self, path: Particle):
        self.cache = path.cache
        path.y = self.response(path.x)
        self.response_calls += 1
        path.score = self.score_function(path)
        path.cache = self.cache

    def update_single_variable(self, path: Particle, variable: int, value: float, lazy: bool = False):
        path.x[variable] = value
        if not lazy:
            self.cache = path.cache
            path.y = self.response(path.x)
            self.response_calls += 1
            path.score = self.score_function(path)
            path.cache = self.cache

    def pdf_conditional_sample(self, path: Particle, score: float) -> Particle:
        new_path = path.copy()
        needs_update = False
        for i in self.scan(random=True):
            xi = self.pdf_marginal_sample(i, 0)
            needs_update = xi > new_path.x[i] or xi > score
            self.update_single_variable(new_path, i, xi, lazy=needs_update)
            if new_path.score < score:
                xi = self.pdf_marginal_sample(i, score)
                self.update_single_variable(new_path, i, xi)
        if needs_update:
            self.update_particle(new_path)
            assert new_path.score >= score
        return new_path


class BinarySystem(FeynmanKac):
    unreliabilities: np.ndarray
    exp_parameters: np.ndarray
    capacities: np.ndarray
    index_calls: int
    search_type: int = 1
    search_neighborhood: bool = True

    class Cache(FeynmanKac.Cache):

        def __init__(self, index: Optional[int] = None, data: Any = None,
                     solution: Optional[Dict] = None):
            self.index = index
            self.data = data
            self.solution = solution

    def sample(self) -> np.ndarray:
        return np.random.exponential(self.exp_parameters)

    def logpdf(self, x: np.ndarray) -> float:
        return (np.log(1 / self.exp_parameters) - 1 / self.exp_parameters * x).sum()

    def pdf(self, x: np.ndarray) -> float:
        return np.exp(self.logpdf(x)).item

    def pdf_marginal_sample(self, variable: int, low: float = 0):
        return np.random.exponential(self.exp_parameters[variable]) + low

    @abc.abstractmethod
    def index_feasible(self, variable: int, x: np.ndarray) -> bool:
        """True if variable repair time is equal or greater than system repair time."""
        pass

    def first_feasible_index(self, x: np.ndarray):
        def binary_search(arr: np.ndarray):
            if len(arr) == 1:
                return arr.item()
            else:
                a, b = np.array_split(arr, 2)
                if self.index_feasible(a[-1], x):
                    return binary_search(a)
                else:
                    return binary_search(b)

        indices = x.argsort()
        first_index = None

        # Search index neighborhood
        if self.search_neighborhood and self.cache.index is not None:
            bound = indices.tolist().index(self.cache.index)
            if self.index_feasible(self.cache.index, x):
                indices = indices[:bound + 1]
                if bound == 0 or not self.index_feasible(indices[bound - 1], x):
                    first_index = self.cache.index
                else:
                    indices = indices[:bound]
            else:
                indices = indices[bound + 1:]

        if first_index is None:
            if self.search_type == 0:
                # Linear search
                for first_index in indices:
                    if self.index_feasible(first_index, x):
                        break
                else:
                    self.index_feasible(first_index, x)
                    raise IOError('No feasible index. Non-monotonic LSF?')
            elif self.search_type == 1:
                # Binary search
                first_index = binary_search(indices)
            else:
                raise NotImplementedError  # search needs to be linear (0) or binary (1)

        if self.cache.index != first_index:
            self.index_feasible(first_index, x)

        return first_index

    def response(self, x: np.ndarray) -> np.ndarray:
        return x[self.first_feasible_index(x)]

    def score_function(self, path: Particle) -> float:
        return path.y.item()

    def proposal_pdf_sample(self, path: Particle, level: int) -> np.ndarray:
        return np.abs(super().proposal_pdf_sample(path, level))
