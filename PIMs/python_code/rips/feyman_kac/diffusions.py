from rips.base import *

import numpy as np
from scipy import stats

__all__ = ['DiffusionOneBarrier', 'DiffusionTwoBarriers']


class Diffusion(FeynmanKac):
    r"""Implements 1-dimesional diffusion

    .. math::
        dX(t) = \sqrt{2\beta^{-1}} d W(t)

    with inital condition :math:`X(0)=0.1$`. The goal is to compute

    .. math::
        \Pr\{\vert X(T)\vert>1\}

    with :math:`T=1`.

    """
    def __init__(self, tf: float = 1.0, dt: float = 1e-3, beta: float = 2, x0: float = 0.1,
                 sampling_method: int = 0):
        self.tf = tf
        self.dt = dt
        self.N = int(tf / dt)
        self.beta = beta
        self.scale = np.sqrt(2 / self.beta * self.dt)
        self.x0 = x0
        self.sampling_method = sampling_method
        self.levels = 0

    def sample_particle(self):
        x = np.random.randn(self.N)
        y = (self.x0 / self.scale + np.sum(x)) * self.scale
        path = Particle(x, y)
        path.score = self.score_function(path)
        return path


class DiffusionOneBarrier(Diffusion):

    def __init__(self, tf: float = 1.0, dt: float = 1e-3, beta: float = 2, x0: float = 0.1,
                 a: float = 1, sampling_method: int = 0):
        self.a = a
        super().__init__(tf=tf, dt=dt, beta=beta, x0=x0, sampling_method=sampling_method)
        self.scores = [-np.inf]

    def log_potential(self, path: Particle):
        return 0 if path.score > 1 else -np.inf

    def score_function(self, path: Particle):
        return path.y / self.a

    def critical_value(self, y, level):
        return - (y - self.scores[level] * self.a) / self.scale

    def kernel(self, path: Particle, level: int):
        path = path.copy()
        x = path.x
        y = path.y
        scan = np.random.rand(len(x)).argsort()
        for i in scan:
            y -= x[i] * self.scale
            xa = self.critical_value(y, level)
            if self.sampling_method == 0:
                xi = np.random.randn()
                while xi < xa if self.x0 < self.a else xi > xa:
                    xi = np.random.randn()
            else:
                if self.x0 < self.a:
                    a = stats.norm.cdf(xa)
                    ui = np.random.uniform(a, 1)
                else:
                    a = stats.norm.cdf(xa)
                    ui = np.random.uniform(0, a)
                xi = stats.norm.ppf(ui)
            x[i] = xi
            y += xi * self.scale
        path = Particle(x, y)
        path.score = self.score_function(path)
        return path


class DiffusionTwoBarriers(Diffusion):

    def __init__(self, tf: float = 1.0, dt: float = 1e-3, beta: float = 2, x0: float = 0.1,
                 a: float = 1, b: float = 1, sampling_method: int = 0):
        self.barrier1 = DiffusionOneBarrier(tf=tf, dt=dt, beta=beta, x0=x0, a=a, sampling_method=sampling_method)
        self.barrier2 = DiffusionOneBarrier(tf=tf, dt=dt, beta=beta, x0=x0, a=b, sampling_method=sampling_method)
        super(DiffusionTwoBarriers, self).__init__(tf=tf, dt=dt, beta=beta, x0=x0, sampling_method=sampling_method)
        self.scores = [-np.inf]
        self.barrier1.scores = self.scores
        self.barrier2.scores = self.scores

    def score_function(self, path: Particle):
        return max(self.barrier1.score_function(path), self.barrier2.score_function(path))

    def kernel(self, path: Particle, level: int):
        if self.barrier1.score_function(path) >= self.scores[level]:
            new_path = self.barrier1.kernel(path, level)
        else:
            assert self.barrier2.score_function(path) >= self.scores[level]
            new_path = self.barrier2.kernel(path, level)
        assert new_path.score >= self.scores[level]
        return new_path
