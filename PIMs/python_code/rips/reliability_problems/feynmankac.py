import abc
from typing import List

import numpy as np
from scipy import stats


class Gaussian(FeynmanKac):

    def __init__(self, mu, sigma, a=-1, b=1):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

    def kernel0(self):
        return stats.norm.rvs(self.mu, self.sigma)

    def kernel(self, x_prev, i):
        return stats.norm.rvs(x_prev, self.sigma)

    def log_potential(self, x):
        return 0 if x < self.a | x > self.b else -np.inf

    def score_function(self, path):
        return min(path.x - self.a, self.b - path.x)






beta = 2
dt = 1
tf = 1
x0 = 0.1
s = 5
num_samples = 10000



# import matplotlib.pyplot as plt
# print(np.mean([fk.kernel(fk.sample_particle(), 0).score > 1 for _ in range(num_samples)]))
# t = [0] + np.cumsum(np.arange(int(tf/dt)) * dt + dt).tolist()
# for i in range(10):
#     plt.plot(t, np.cumsum([fk.x0] + fk.sample_particle().x.tolist()) * fk.scale)
# for i in range(10):
#     plt.plot(range(fk.N), np.cumsum(fk.kernel(fk.sample_particle(), 0).x))
# plt.show()
#
# gs_pilot(fk, s)
# print(fk.scores)
# print(np.mean([gs(fk, s) for i in range(num_samples)]))

a = 1
b = -1

fk = DiffusionOneBarrier(a=a, beta=beta, dt=dt, tf=tf, x0=x0)
print(np.mean([fk.sample_particle().score > 1 for _ in range(num_samples)]))
print(np.mean([fk.kernel(fk.sample_particle(), 0).score > 1 for _ in range(num_samples)]))

# for i in range(1, 7):
# for i in []:
#     beta = 2 ** i
#     fk = DiffusionTwoBarriers(a=a, b=b, beta=beta, dt=dt, tf=tf, x0=x0)
    # gs_pilot(fk, s=s)
    # approx = np.mean([gs(fk, s) for i in range(num_samples)])
    # approx = smc(fk, num_samples, s)
    # print(fk.scores)
    # exact = stats.norm.cdf(b, fk.x0, np.sqrt(2/beta)) + (1 - stats.norm.cdf(a, fk.x0, np.sqrt(2/beta)))
    # ci = smc_confidence_intervals(approx, num_samples, s)
    # print("{:.4e} {:.4e} {:.3f} {:.4e} {:.4e} {:.4e}".format(exact, approx, approx / exact - 1, ci[0], exact, ci[1]))