from time import perf_counter
from typing import Callable

import numpy as np

from rips.utils import *
from rips.rare_events.generalized_splitting import *
from rips.rare_events.sequential_mc import *

__all__ = ['SMC', 'AlgoIPS', 'ComboIPS', 'ComboUQ', 'ComboUQ2']

from rips.utils.kernel import Kernels


class AlgoIPS:
    """Base class for running IPS algorithms."""
    mu: float
    """Sample mean."""
    ci: np.ndarray
    """Sample confidence intervals"""
    s_factor: int = 2
    """Splitting factor."""

    def best_variance(self, p: float) -> float:
        """Returns best possible variance with an idealized kernel.

        Args:
            p: exact probability.
        """
        p0 = 1 / self.s_factor
        n0 = np.floor(np.log(p) / np.log(p0)).item()
        r0 = p * p0 ** -n0
        return p ** 2 * (n0 * (1 - p0) / p0 + (1 - r0) / r0)

    def bias(self, p: float) -> float:
        """Returns relative variance for adaptive level methods.

        Args:
            p: exact probability.
        """
        pass

    def sample_size(self, p: float, epsilon: float = 0.1, delta: float = 0.95):
        """Returns idealized sample size assuming CLT.

        Args:
            p: exact probability.
            epsilon: maximum relative error.
            delta: probability of exceeding error.
        """
        from scipy.stats import norm
        sd = np.sqrt(self.best_variance(p) / p ** 2)

        def test(n):
            return 2 * norm.cdf(-epsilon, scale=sd / np.sqrt(n)) < delta

        return galloping_search(2, test)
        # relative_variance = self.best_variance(p) / p ** 2
        # return int(np.ceil(2 * relative_variance / epsilon ** 2 * np.log(1 / (np.sqrt(2 * np.pi) * delta))))

    def naive_sample_size(self, p: float, epsilon: float = 0.1, delta: float = 0.95):
        """Returns worst-case sample size assuming CLT.

        Args:
            p: exact probability.
            epsilon: maximum relative error.
            delta: probability of exceeding error.

        """
        from scipy.stats import norm
        sd = np.sqrt((p * (1 - p)) / p ** 2)

        def test(n):
            return 2 * norm.cdf(-epsilon, scale=sd / np.sqrt(n)) < delta

        return galloping_search(2, test)
        # relative_variance = p * (1 - p) / p ** 2
        # return int(np.ceil(2 * relative_variance / epsilon ** 2 * np.log(2 / delta)))

    def stats(self):
        """ Returns summary line with results.
        """
        return '{:.4e},{:.4e},{:.4e}'.format(self.mu, *self.ci)


class SMC(AlgoIPS):

    def __init__(self, fk: FeynmanKac, num_particles=int(1e3), s_factor: int = 2, burn: int = 1):
        self.num_particles = num_particles
        self.s_factor = s_factor
        self.burn = burn
        self.mu = smc(fk)
        self.ci = self.confidence_intervals(self.mu, self.num_particles)
        self.var = self.best_variance(self.mu)

    def confidence_intervals(self, p, n):
        p0 = 1 / self.s_factor
        n0 = np.floor(np.log(p) / np.log(p0))
        r0 = p * p0 ** -n0
        return p * (1 - n0 * (1 - p0) / p0 / n + np.array([-1.0, 1.0]) * 2 * np.sqrt(
            (n0 * (1 - p0) / p0 + (1 - r0) / r0) / n))

    def sample_variance(self, fk: FeynmanKac, num_samples=100) -> float:
        sample = [smc(fk) for _ in range(num_samples)]
        return np.var(sample, ddof=1).item()


# class GS(AlgoIPS):
#
#     def __init__(self, fk: FeynmanKac, num_particles=int(1e3), s_factor: int = 2, burn: int = 1):
#         self.num_particles = num_particles
#         self.s_factor = s_factor
#         self.burn = burn
#         if fk.levels == 0:
#             smc(fk)
#         self.sample = [gs(fk) for _ in range(self.num_particles)]
#
#     @functools.cached_property
#     def mu(self):
#         return np.mean(self.sample).item()
#
#     @functools.cached_property
#     def std(self):
#         return np.std(self.sample, ddof=1)
#
#     @functools.cached_property
#     def ci(self):
#         return self.mu + 1.96 * np.array([-1, 1]) * self.std / np.sqrt(self.num_particles)
#
#     @functools.cached_property
#     def var(self):
#         return np.var(self.sample, ddof=1)
#
#     @functools.cached_property
#     def cov(self):
#         return self.std / self.mu / np.sqrt(self.num_particles)


class ComboIPS(AlgoIPS):

    def __init__(self, fk: FeynmanKac, p: float = np.nan, verbose: int = 1):
        self.num_particles = fk.num_of_particles
        self.s_factor = fk.s_factor
        # Adaptive levels
        self.time_smc = - perf_counter()
        self.p_smc = smc(fk)
        self.time_smc += perf_counter()
        if verbose:
            print('done smc')
        # Fixed levels
        self.time = -perf_counter()
        self.sample = [gs(fk) for _ in range(self.num_particles)]
        self.time += perf_counter()
        if verbose:
            print('done gs')
        self.p = p if p != np.nan else self.mu
        if verbose:
            print(self.stats_line())
            fk.kernel_stats()
        self.mean = np.mean(self.sample).item()
        self.std = np.std(self.sample, ddof=1).item()

    @property
    def ci(self):
        variance = max(self.var, self.best_variance(self.mu), self.best_variance(self.p))
        return self.mu + 2 * np.array([-1, 1]) * np.sqrt(variance / self.num_particles)

    @property
    def var(self) -> float:
        return np.var(self.sample, ddof=1).item()

    @property
    def cov(self):
        return self.std / self.p / np.sqrt(self.num_particles)

    def ci_smc(self, ideal: bool = False):
        n = self.num_particles
        p0 = 1 / self.s_factor
        n0 = np.floor(np.log(self.p) / np.log(p0))
        if ideal:
            variance = self.best_variance(self.p)
        else:
            variance = max(self.var, self.best_variance(self.p))
        return self.p_smc * (1 - n0 * (1 - p0) / p0 / n) + np.array([-1.0, 1.0]) * 2 * np.sqrt(variance / n)

    def uncertainty(self, ci):
        exponent = np.floor(np.log10(np.abs(self.p))).astype(int)
        mu = (ci[1] + ci[0]) / 2 * 10 ** - exponent
        dev = (ci[1] - ci[0]) / 2 * 10 ** (-exponent + 2)
        return f'{mu:.2f}({dev:3.0f})e{exponent}'

    def uncertainty_smc(self):
        variance = max(self.var, self.best_variance(self.p))
        return 2 * np.sqrt(variance / self.num_particles)

    header = ', '.join(['p', 'mu', 'var', 'num_of_particles', 'time', 'variance_ratio', 'cov',
                        'p_smc', 'time_smc',
                        'cil', 'cih', 'cil_smc', 'cih_smc'])

    def stats_line(self):
        p_smc = self.uncertainty(self.ci_smc())
        p_gs = self.uncertainty(self.ci)
        variance_ratio = self.best_variance(self.p) / self.var
        line = f'{self.p:.2e}, {p_gs}, {self.var:.5e}, {self.num_particles}, {self.time:.4f}, '
        line += f'{variance_ratio:.5f}, {self.cov:.5f}, {p_smc}, {self.time_smc:.4f}, '
        line += ', '.join([f'{val:.4e}' for val in (*self.ci, *self.ci_smc(ideal=False))])
        return line


def binary_search(lb: int, ub: int, test: Callable):
    if lb == ub:
        return lb

    mid = int(np.floor((lb + ub) / 2))

    if test(mid):
        return binary_search(lb, mid, test)
    else:
        return binary_search(mid + 1, ub, test)


def galloping_search(lb: int, test: Callable):
    ub = 2 * lb
    while not test(ub):
        lb = ub
        ub *= 2

    return binary_search(lb, ub, test)


from typing import List, Dict


class ComboUQ(AlgoIPS):

    def __init__(
            self,
            fk: FeynmanKac,
            kernels: Kernels,
            cloud_name: str = None,
            cloud_smc_name: str = None,
            lineage: bool = False,
            lineage_smc: bool = False,
            verbose: int = 1,
    ):
        self.fk = fk
        self.kernels = kernels
        self.cloud = Cloud(file_name=cloud_name)
        self.cloud_smc = Cloud(file_name=cloud_smc_name)
        self.lineage = lineage
        self.lineage_smc = lineage_smc
        # adaptive levels
        if verbose > 0: print('Adaptive levels...')
        self.cpu_time_smc = - perf_counter()
        self.p_smc = self.adaptive_levels()
        self.cpu_time_smc += perf_counter()
        self.ng_smc = fk.total_response_calls
        # fixed levels
        if verbose > 0: print('Fixed levels...')
        self.cpu_time_gs = - perf_counter()
        self.samples = self.fixed_levels()
        self.p_bar = np.mean(self.samples).item()
        self.cpu_time_gs += perf_counter()
        self.ng_gs = fk.total_response_calls - self.ng_smc
        self.k = fk.levels
        # call results for logs
        self.var = np.var(self.samples, ddof=1).item()
        self.std = np.std(self.samples, ddof=1).item()
        # self.cov = self.std / self.p_bar
        # self.cov_smc = self.std / self.p_smc
        if verbose > 0: print('done')

    def adaptive_levels(self) -> float:
        if not self.cloud_smc.particles:
            psmc = smc(self.fk, self.kernels, cloud=self.cloud_smc,
                       lineage=self.lineage_smc)
        else:
            arr = [(x.score >= 1) / self.fk.s_factor ** x.k for x in self.cloud_smc.get_leafs]
            psmc = np.sum(arr) / self.fk.num_of_particles
        return psmc

    def fixed_levels(self) -> List[float]:
        if not self.cloud.particles:
            samples = [gs(self.fk, self.kernels, cloud=self.cloud, lineage=self.lineage) for _ in range(self.fk.num_of_particles)]
        else:
            samples = [
                np.sum([(x.score >= 1) / self.fk.s_factor ** x.k for x in seed.leaves()])
                for seed in self.cloud.particles
            ]

        return samples

    def fixed_levels_update(self):
        self.cpu_time_gs = - perf_counter()
        self.samples = [gs(self.fk, self.kernels, cloud=self.cloud) for _ in range(self.fk.num_of_particles)]
        self.p_bar = np.mean(self.samples).item()
        self.cpu_time_gs += perf_counter()
        # call results for logs
        self.var = np.var(self.samples, ddof=1).item()
        self.std = np.std(self.samples, ddof=1).item()

    def confidence_intervals(self):
        return self.p_bar + 1.96 * self.std / np.sqrt(self.fk.num_of_particles) * np.array([-1, 1])

    def save_results(self):
        self.cloud_smc.to_file()
        self.cloud.to_file()

    def load_results(self):
        if self.cloud_smc is not None:
            self.cloud_smc.from_file()

    def summary_results(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if type(v) in {float, int, (float, float), np.float64}}


class ComboUQ2(AlgoIPS): # aE-SuS + aPIM

    def __init__(
            self,
            fk: FeynmanKac,
            kernels: Kernels,
            cloud_name: str = None,
            lineage: bool = False,
    ):
        
        self.fk      = fk
        self.kernels = kernels
        self.cloud   = Cloud(file_name=cloud_name)
        self.lineage = lineage
        
        # fixed levels
        self.cpu_time_gs = - perf_counter()
        self.samples = self.fixed_levels()
        self.p_bar = np.mean(self.samples).item()
        self.cpu_time_gs += perf_counter()
        self.ng_gs = fk.total_response_calls
        self.k     = fk.levels
        # call results for logs
        self.var = np.var(self.samples, ddof=1).item()
        self.std = np.std(self.samples, ddof=1).item()

    def adaptive_levels(self) -> float:
        if not self.cloud_smc.particles:
            psmc = smc(self.fk, self.kernels, cloud=self.cloud_smc,
                       lineage=self.lineage_smc)
        else:
            arr = [(x.score >= 1) / self.fk.s_factor ** x.k for x in self.cloud_smc.get_leafs]
            psmc = np.sum(arr) / self.fk.num_of_particles
        return psmc

    def fixed_levels(self) -> List[float]:
        if not self.cloud.particles:
            samples = [gs(self.fk, self.kernels, cloud=self.cloud, lineage=self.lineage) for _ in range(self.fk.num_of_particles)]
        else:
            samples = [
                np.sum([(x.score >= 1) / self.fk.s_factor ** x.k for x in seed.leaves()])
                for seed in self.cloud.particles
            ]

        return samples

    def fixed_levels_update(self):
        self.cpu_time_gs = - perf_counter()
        self.samples = [gs(self.fk, self.kernels, cloud=self.cloud) for _ in range(self.fk.num_of_particles)]
        self.p_bar = np.mean(self.samples).item()
        self.cpu_time_gs += perf_counter()
        # call results for logs
        self.var = np.var(self.samples, ddof=1).item()
        self.std = np.std(self.samples, ddof=1).item()

    def confidence_intervals(self):
        return self.p_bar + 1.96 * self.std / np.sqrt(self.fk.num_of_particles) * np.array([-1, 1])

    def save_results(self):
        self.cloud_smc.to_file()
        self.cloud.to_file()

    def load_results(self):
        if self.cloud_smc is not None:
            self.cloud_smc.from_file()

    def summary_results(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if type(v) in {float, int, (float, float), np.float64}}