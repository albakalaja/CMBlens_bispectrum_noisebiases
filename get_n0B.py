"""
Run this module to compute N^0_B as in eq. (3.12) of https://arxiv.org/abs/2210.16203 in the equilateral and folded configurations. 
Equilateral takes < 1 min with the following setup on 64 cpu. 
"""
import numpy as np
import os, sys
from pathos.multiprocessing import ProcessingPool as Pool

import scripts.N0B_bias as N0B # module for the N0 bias term

# CMB ells
rlmin, rlmax = 2, 2000
rsample = 100 # or fully sampled: int(rlmax-rlmin+1)
l1 = np.linspace(rlmin, rlmax, rsample)

N_phi = 50 # vegas will take many more points
phi1 = np.linspace(0., 2*np.pi, N_phi) # angle between l1 and L

# Reconstruction Ls
Lmin, Lmax = 2, 2000
L = np.linspace(Lmin,Lmax+1,rsample) # or fully sampled: int(Lmax-Lmin+1)


# Load cls
ell = np.arange(0,8000+1,1)

cls_path = './cls'
glclTT = np.loadtxt(f'{cls_path}/glensed_clTT_lmax8000.txt') # we use gradient lensed Cls
lclTT = np.loadtxt(f'{cls_path}/lensed_clTT_lmax8000.txt')

# possible configurations: equilateral and folded, squeezed has been implemented but not tested yet.
def get_equil(L):
    return N0B.N0_total(L, l1, phi1, ell, glclTT, lclTT, lclTT, rlmin, rlmax, 1e5, 'equil', True)
def get_fold(L):
    return N0B.N0_total(L, l1, phi1, ell, glclTT, lclTT, lclTT, rlmin, rlmax, 1e5, 'fold', True)


data = np.zeros((len(L), 2))
data[:, 0] = np.copy(L)

pool = Pool(ncpus=int(os.getenv('OMP_NUM_THREADS')))
data[:, 1] = np.array(pool.map(get_equil, L)).flatten()

np.savetxt(f'./output/N0equil_rlmax{rlmax}Lmax{Lmax}_gCl_lCl_lCl.txt', data)

