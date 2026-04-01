# %% preamble
import numpy as np
import math
import scipy.io
from scipy.stats import norm
import multiprocessing as mp
#import matlab.engine
import pickle
import oct2py
import sys

from rips import *
from rips.feyman_kac import StandardGaussian
from rips.utils import FeynmanKac, Particle

# %% settings for the problem

caseName        = 'case14'
blackoutSizeThr = 54.8
pf              = 1.1373*10**-4

method          = 'iPIM_aPIM'
s_factor        = 10
N               = 2000
nRun            = 200
batchSize       = 10

# %% define the DC optimal power flow class
class dcopt(FeynmanKac, StandardGaussian):
    def __init__(self, thr: float, mpc: dict, eng, **kwargs):
        self.eng   = eng  # MATLAB API
        self.thr   = thr
        self.alpha = 2.0
        self.mpc   = eng.add_branchCapacity(mpc, self.alpha)

        self.set_probablisticModel(**kwargs)
        self.get_networkInfo()

        super().__init__(**kwargs)

    def set_probablisticModel(self, **kwargs):
        if 'distrGen' in kwargs:
            self.distrGen = kwargs['distrGen']
        else:
            self.distrGen = [[1, 0.6, 0.2, 0], [0.01, 0.2, 0.5, 1]]
        if 'distrOrdinaryBus' in kwargs:
            self.distrOrdinaryBus = kwargs['distrOrdinaryBus']
        else:
            self.distrOrdinaryBus = [[1, 0], [0.01, 1]]
        if 'distrBranch' in kwargs:
            self.distrBranch = kwargs['distrBranch']
        else:
            self.distrBranch = [[1, 0], [0.01, 1]]

    def get_networkInfo(self):
        self.nb  = len(self.mpc['bus'])    # number of buses
        self.ng  = len(self.mpc['gen'])    # number of generators
        self.nl  = len(self.mpc['branch']) # number of branches
        self.busDic    = [ int(self.mpc['bus'][i][0]) for i in range(self.nb) ]
        self.genDic    = [ int(self.mpc['gen'][i][0]) for i in range(self.ng) ]
        self.branchDic = [ [int(self.mpc['branch'][i][0]), int(self.mpc['branch'][i][1])] for i in range(self.nl) ]

    # define the respone function that is the system performance
    def response(self, path: np.ndarray, level: int) -> np.ndarray:

        systemState = []
        for d in range(self.nb):
            if self.busDic[d] in self.genDic:
                idx = np.argwhere( norm.cdf(path[d]) <= self.distrGen[1] )
                systemState.append( [self.distrGen[0][ int(idx[0][0]) ]] )
            else:
                idx = np.argwhere( norm.cdf(path[d]) <= self.distrOrdinaryBus[1] )
                systemState.append( [self.distrOrdinaryBus[0][ int(idx[0][0]) ]] )

        for d in range(self.nb, self.nb + self.nl):
            idx = np.argwhere( norm.cdf(path[d]) <= self.distrBranch[1] )
            systemState.append( [self.distrBranch[0][ int(idx[0][0]) ]] )

        # eng.workspace['mpc'] = self.mpc
        # eng.workspace['systemState'] = matlab.double(systemState)
        # lackoutSize = eng.func_dcopt(eng.workspace['systemState'], eng.workspace['mpc'] )
        blackoutSize = self.eng.func_dcopt(matlab.double(systemState), self.mpc)

        return blackoutSize

    def score_function(self, particle: Particle) -> float:
        return particle.response / self.thr

    @property
    def num_variables(self) -> int:
        return self.nb + self.nl


# %% function for parallel
def iPIM_aPIM(iRun, octave):

    #eng = matlab.engine.start_matlab()
    #mpc0 = eng.loadcase(caseName)
    mpc0 = octave.eval(caseName)

    if 'bus_name' in mpc0.keys():
        del mpc0['bus_name']

    model = dcopt( thr = blackoutSizeThr, mpc = mpc0, eng = eng)
    model.num_of_particles = N
    model.s_factor         = s_factor

    kernels = [PCN()]

    results = ComboUQ(model, kernels)

    eng.quit()
    print(iRun)

    return results.summary_results()

# %% parallel run of the PIM

def main(octave):

    nGroup = math.ceil( nRun/batchSize )

    for iGroup in range(nGroup):

        if iGroup == nGroup-1:
            nRun_i = nRun - (nGroup-1)*batchSize
        else:
            nRun_i = batchSize

        fileName = 'batch' + str(iGroup) + '_' + caseName + '_thr' + str(blackoutSizeThr) + '_iPIM+aPIM_N' + str(N)
        with mp.Pool(3) as my_pool:
            results = my_pool.map(iPIM_aPIM, [i for i in range(iGroup*batchSize, iGroup*batchSize+nRun_i)])

        estpf_officialRun   = [results[i]['p_bar'] for i in range(nRun_i)]
        estpf_pilotRun      = [results[i]['p_smc'] for i in range(nRun_i)]
        numOfParticles_aPIM = [results[i]['ng_gs'] for i in range(nRun_i)]
        numOfParticles_iPIM = [results[i]['ng_smc'] for i in range(nRun_i)]

        with open(fileName, 'wb') as fid:
            pickle.dump([estpf_officialRun, estpf_pilotRun, numOfParticles_aPIM, numOfParticles_iPIM], fid)


