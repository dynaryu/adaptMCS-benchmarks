from itertools import count
from typing import Callable, List

import numpy as np

__all__ = [
    # Types
    'Path', 'Response', 'Score', 'ScoreList', 'Level', 'Particles',
    # Classes
    'Particle',
]

Path = np.ndarray
"""Sample path type."""
Response = np.ndarray
"""Response type."""
Score = float
"""Score type."""
ScoreList = List[Score]
"""List of scores."""
Level = int
"""Specifies transition."""


class Particle:
    """Holds data of particle."""

    def __init__(
            self,
            path: Path,
            response: Response,
            score: Score = None,
            number: int = None,
            origin: int = None,
            u: float = None,
            k: int = 0,
            predecessor: 'Particle' = None,
            keep_lineage: bool = False
    ):
        self.path = path
        """Outcome of random process."""
        self.response = response
        """Response due to random outcome."""
        self.score = score
        """Score of particle."""
        self.number = number
        """Identifies particle in given level."""
        self.origin = origin
        """Identifies origin particle from previous level."""
        self.predecessor = predecessor
        """Parent particle."""
        self.successors: List[Particle] = []
        """List of successor particles."""
        self.k = k
        """MCMC index."""
        # self.u = np.random.rand() if u is None else u
        self.u = None
        """Auxiliary estimation var."""
        self.keep_lineage: bool = keep_lineage
        """LSF value"""
        self.lsf = None

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return "Particle(score={:.3f})".format(self.score)

    def copy(self) -> 'Particle':
        """Safe copy of attributes except for cache."""
        path = self.path + 0
        response = self.response + 0
        score = self.score + 0
        keep_lineage = self.keep_lineage
        particle = Particle(
            path=path,
            response=response,
            score=score,
            u=self.u,
            keep_lineage=keep_lineage
        )
        return particle

    def recursive_apply(self, f: Callable):
        """Recursively apply function.

         Returns average between successors.
         """
        s = len(self.successors)
        if s == 0:
            return f(self)
        else:
            return np.mean([successor.recursive_apply(f) for successor in
                            self.successors]).item()

    def leaves(self):
        if not self.successors:
            yield self
        else:
            for part in self.successors:
                yield from part.leaves()

    def to_graph(self):
        c = count(0)

        def arcs_it(p: Particle, vp: int):
            for s in p.successors:
                vs = next(c)
                yield (vp, vs)
                yield from arcs_it(s, vs)

        def nodes_it(p: Particle):
            yield p
            for s in p.successors:
                yield from nodes_it(s)

        nodes = list(nodes_it(self))
        arcs = list(arcs_it(self, next(c)))
        return nodes, arcs

    def lineage(self, parent: 'Particle'):
        parent.successors.append(self)
        self.predecessor = parent
        self.k = parent.k + 1


Particles = List[Particle]
