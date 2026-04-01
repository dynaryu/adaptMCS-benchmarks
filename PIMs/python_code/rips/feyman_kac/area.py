import numpy as np

from rips.base import *


class Area(FeynmanKac):

    def __init__(self, regions, d: int = 2):
        self.regions = regions
        self.d = d

    def sample_particle(self) -> Particle:
        x = np.random.rand(self.d)
        y = self.regions[0].contains_point(x)
        score = y
        path = Particle(x, y, score)
        return path

    def kernel(self, path: Particle, i) -> Particle:
        dx = 1/12
        path = path.copy()
        x = path.x
        y = path.y
        scan = np.random.rand(len(x)).argsort()
        for i in scan:
            xi = np.random.uniform(x[i] - dx, x[i] + dx)


