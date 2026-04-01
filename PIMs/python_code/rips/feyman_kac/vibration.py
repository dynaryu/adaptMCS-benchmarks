import numpy as np
from scipy import signal
from scipy.stats import norm

from rips.base import *

__all__ = ['VibrationFirstPassage', 'VibrationTopBarrier', 'VibrationBottomBarrier']


class Vibration(FeynmanKac):

    def __init__(self, omega, zeta, tf, s=1, t0=0, dt=0.02):
        self.omega = omega
        self.omega_d = omega * np.sqrt(1 - zeta ** 2)
        self.zeta = zeta
        self.s = s
        self.dt = dt
        self.nt = int((tf - t0) / dt) + 1
        self.t = np.linspace(t0, tf, self.nt)
        self.scale = np.sqrt(2 * np.pi * self.s * self.dt)
        self.h = self.impulse_response()
        self.h_lag = np.array([self.impulse_response_lag(i) for i in range(self.nt)]) * self.scale

    def impulse_response(self, tau=0):
        h = np.exp(-self.omega * self.zeta * self.t) / self.omega_d * np.sin(self.omega_d * self.t)
        h[tau: self.nt] = h[: self.nt - tau]
        h[:tau] = 0
        return h

    def impulse_response_lag(self, tau=0):
        h = self.h + 0
        h[tau: self.nt] = h[: self.nt - tau]
        h[:tau] = 0
        return h

    def response(self, x):
        return np.dot(x, self.h_lag)

    def variance(self, t):
        a = self.omega * self.zeta
        c = self.omega_d
        s = 2 * np.pi * np.exp(- 2 * a * t)
        s *= (a ** 2 * np.cos(2 * c * t) - a ** 2 + c ** 2 * (np.exp(2 * a * t) - 1) - a * c * np.sin(2 * c * t))
        s /= 4 * a * c ** 2 * (a ** 2 + c ** 2)
        return s

    def ss_representation(self):
        a = np.array([[0, 1], [-self.omega ** 2, - 2 * self.zeta * self.omega]])
        b = np.array([[0], [1]])
        c = np.array([1, 0])
        d = np.array([0])
        return signal.StateSpace(a, b, c, d)

    def sys_output(self, x):
        sys = self.ss_representation()
        return signal.lsim(sys, x * self.scale / self.dt, self.t)


class VibrationTopBarrier(Vibration):
    def __init__(self, omega, zeta, a, tf, s=1, t0=0, dt=0.02):
        super().__init__(omega=omega, zeta=zeta, tf=tf, s=s, t0=t0, dt=dt)
        self.a = a
        self.num_var = self.nt
        self.scores = [-np.inf]
        self.levels = 0

    def score_function(self, path: Particle) -> float:
        return path.y.max() / self.a

    def kernel(self, path: Particle, level) -> Particle:
        path = path.copy()
        x = path.x
        y = path.y
        for i in self.scan():
            h = self.h_lag[i]
            y -= h * x[i]
            xi = np.random.randn()
            if (y + h * xi).max() < self.scores[level] * self.a:
                xmin, xmax = self.critical_value(y, i, level)
                umin = 1 - norm.cdf(xmin)
                umax = norm.cdf(xmax)
                gap = 1 - umin - umax
                u = np.random.uniform(0, umin + umax)
                if u < umax:
                    xi = norm.ppf(u)
                elif u <= (umin + umax):
                    xi = norm.ppf(u + gap)
                else:
                    raise ValueError
            x[i] = xi
            y += h * xi
        path.score = self.score_function(path)
        return path

    def critical_value(self, y, i, level):
        h = self.h_lag[i]
        constraints = (self.scores[level] * self.a - y) / h
        xmin = constraints[h >= 0].min()
        xmax = cs.max() if len(cs := constraints[h < 0]) > 0 else -np.inf
        return xmin, xmax

    def sample_particle(self) -> Particle:
        x = np.random.randn(self.nt)
        y = self.response(x)
        path = Particle(x, y)
        path.score = self.score_function(path)
        return path


class VibrationBottomBarrier(Vibration):
    def __init__(self, omega, zeta, b, tf, s=1, t0=0, dt=0.02):
        super().__init__(omega=omega, zeta=zeta, tf=tf, s=s, t0=t0, dt=dt)
        self.b = b
        self.num_var = self.nt
        self.scores = [-np.inf]
        self.levels = 0

    def score_function(self, path: Particle) -> float:
        return - path.y.min() / self.b

    def kernel(self, path: Particle, level) -> Particle:
        path = path.copy()
        x = path.x
        y = path.y
        for i in self.scan():
            h = self.h_lag[i]
            y -= h * x[i]
            xi = np.random.randn()
            if (y + h * xi).min() > - self.scores[level] * self.b:
                xmin, xmax = self.critical_value(y, i, level)
                umin = 1 - norm.cdf(xmin)
                umax = norm.cdf(xmax)
                gap = 1 - umin - umax
                u = np.random.uniform(0, umin + umax)
                if u < umax:
                    xi = norm.ppf(u)
                elif u <= (umin + umax):
                    xi = norm.ppf(u + gap)
                else:
                    raise ValueError
            x[i] = xi
            y += h * xi
        path.score = self.score_function(path)
        return path

    def critical_value(self, y, i, level):
        h = self.h_lag[i]
        constraints = (self.scores[level] * self.b + y) / h
        xmin = constraints[h >= 0].min()
        xmax = cs.max() if len(cs := constraints[h < 0]) > 0 else -np.inf
        return xmin, xmax

    def sample_particle(self) -> Particle:
        x = np.random.randn(self.nt)
        y = self.response(x)
        path = Particle(x, y)
        path.score = self.score_function(path)
        return path

class VibrationFirstPassage(Vibration):
    def __init__(self, omega, zeta, a, tf, s=1, t0=0, dt=0.02):
        super().__init__(omega=omega, zeta=zeta, tf=tf, s=s, t0=t0, dt=dt)
        self.a = a
        self.num_var = self.nt
        self.scores = [0]
        self.levels = 0

    def score_function(self, path: Particle) -> float:
        return np.abs(path.y).max() / self.a

    def kernel(self, path: Particle, level) -> Particle:
        path = path.copy()
        x = path.x
        y = path.y
        for i in self.scan():
            y -= self.h_lag[i] * x[i]
            xi = np.random.randn()
            while np.abs(y + self.h_lag[i] * xi).max() < self.scores[level] * self.a:
                xi = np.random.randn()
            x[i] = xi
            y += self.h_lag[i] * x[i]
        path.score = self.score_function(path)
        return path

    def sample_particle(self) -> Particle:
        x = np.random.randn(self.nt)
        y = self.response(x)
        path = Particle(x, y)
        path.score = self.score_function(path)
        return path
