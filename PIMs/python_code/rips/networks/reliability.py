from rips.utils import *
from rips.networks.graphs import *
from rips.feyman_kac.gaussian import StandardGaussian
from typing import Dict, Any, List, Type
import numpy as np
from scipy.stats import norm
import abc


__all__ = ['NetRel', 'GraphRel', 'ExpTime', 'GaussTime']


Index: Type[int] = int
"""Index or order statistic."""


def _split(arr):
    n = len(arr)
    center = n // 2 + n % 2
    return arr[:center], arr[center::]


class RepairTime(FeynmanKac):
    """Base class for coherent systems with binary-components.
    """

    def __init__(self, unreliabilities: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.unreliabilities: np.ndarray = unreliabilities
        """Component failure probabilities."""

    # Search
    binary_search: bool = True
    """If true (false), uses binary (linear) search for first feasible index."""
    neighborhood_search: bool = False
    """If true, left and right neighbors of cached index are tried first."""
    num_neighbors: int = 2
    """Number of index neighbors."""

    def feasible_index(self, index: int, path: Path) -> bool:
        """Returns True if the index is equal or greater than the repair-time index.
        """
        raise NotImplementedError

    @property
    def num_variables(self) -> int:
        return len(self.unreliabilities)

    def repair_index(self, path: Path, level: Level) -> Index:
        """Returns index order statistic equal to system repair time.

        Args:
            path: simulation.
        """
        def binary_search(arr: np.ndarray):
            if len(arr) == 1:
                return arr.item()
            else:
                a, b = _split(arr)
                self.response_calls[level] += 1
                if self.feasible_index(a[-1], path):
                    return binary_search(a)
                else:
                    return binary_search(b)

        def linear_search(arr: np.ndarray):
            for j in arr:
                if self.feasible_index(j, path):
                    return j
            else:
                raise IOError('No feasible index.')

        indices = path.argsort()

        # Search
        if self.binary_search:
            first_index = binary_search(indices)
        else:
            # Binary search
            first_index = linear_search(indices)
        # print(first_index)
        return first_index

    def response(self, path: Path, level: Level) -> Response:
        index = self.repair_index(path, level)
        return np.array(path[index])

    def score_function(self, particle: Particle) -> float:
        return particle.response.item()

    def monotonic(self, path: Path) -> bool:
        indices = path.argsort()
        last = False
        for j in indices:
            new = self.feasible_index(j, path)
            if last > new:
                print('Here')
                return False
            else:
                last = new
        return True


class ExpTime(RepairTime):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = 1 - norm.ppf(1 - self.unreliabilities)

    def sample(self) -> Path:
        return np.random.exponential(self.parameters)

    def logpdf(self, path: Path) -> float:
        return (np.log(1 / self.parameters) - 1 / self.parameters * path).sum()


class GaussTime(RepairTime, StandardGaussian):
    """Base class for network reliability."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = 1 - norm.ppf(1 - self.unreliabilities)

    def u_to_x(self, path: Path):
        return path + self.parameters

    def response(self, path: Path, level: Level) -> Response:
        return super().response(self.u_to_x(path), level)


class GaussTime2(RepairTime, StandardGaussian):
    """Base class for network reliability."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameters = norm.ppf(1 - self.unreliabilities)

    def u_to_x(self, path: Path):
        return path / self.parameters

    def response(self, path: Path, level: Level) -> Response:
        return super().response(self.u_to_x(path), level)


class GraphRel(GaussTime, Graph):

    def __init__(self, terminals, **kwargs):
        self.terminals: VertexList = terminals
        """List of terminal nodes."""
        self.latent_calls = 0
        self._source = terminals[0]
        super().__init__(**kwargs)

    def feasible_index(self, index: int, path: Path):
        edge_up = path <= path[index]
        visited = self.visitor(self._source, edge_up)
        self.latent_calls += 1
        for terminal in self.terminals:
            if not visited[terminal]:
                return False
        return True


class NetRel(GaussTime, DiGraph):

    def __init__(self, source, terminals, **kwargs):
        self.source: Node = source
        """Source node."""
        self.terminals: NodeList = terminals
        """List of terminal nodes."""
        super().__init__(**kwargs)

    def feasible_index(self, index: int, path: Path):
        arc_up = path <= path[index]
        visited = self.visitor(self.source, arc_up)
        for terminal in self.terminals:
            if not visited[terminal]:
                return False
        return True
