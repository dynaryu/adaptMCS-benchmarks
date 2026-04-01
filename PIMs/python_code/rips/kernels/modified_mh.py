import numpy as np
from rips.utils import *
from rips.kernels.metropolis import Metropolis


__all__ = ['MMH']


class MMH(Metropolis):

    @property
    def kernel_name(self) -> str:
        return 'mmh'

    def proposal_sample(self, fk: FeynmanKac, path: Path, level: Level) -> Path:
        path_new = path + 2 * np.random.rand(fk.num_variables) - 1
        return path_new

    @staticmethod
    def log_acceptance_rates(fk: FeynmanKac, path: Path, path_new: Path, level: int = 0) -> np.ndarray:
        logrates = fk.logmarginals(path_new) - fk.logmarginals(path)
        return logrates

    def sample_transition(self, fk: FeynmanKac, particle: Particle, level: Level, score: Score) -> Particle:
        path = particle.path
        path_new = self.proposal_sample(fk, path, level)
        logrates = self.log_acceptance_rates(fk, path, path_new)
        accepts = np.random.rand(fk.num_variables) <= np.exp(logrates)
        if any(accepts):
            path_new = path * np.logical_not(accepts) + path_new * accepts
            particle_new = fk.compute_particle(path_new, level)
            if particle_new.score >= score:
                return particle_new
        return particle
