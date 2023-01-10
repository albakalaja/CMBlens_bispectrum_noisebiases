"""
Run this module to compute N^1_B as in eqs. (4.2)-(4.3) of https://arxiv.org/abs/2210.16203 in the equilateral and folded configurations. 
Separable + Equilateral takes < 1 min with the following setup on 64 cpu. 
Coupled + Equilateral takes < 1 min with the following setup on 64 cpu. 

"""
import numpy as np
import os, sys
from pathos.multiprocessing import ProcessingPool as Pool

import scripts.N1B_bias as N1B # module for the N0 bias term

# CMB ells
rlmin, rlmax = 2, 2000
rsample = 100 # or fully sampled: int(rlmax-rlmin+1)
l1 = np.linspace(rlmin, rlmax, rsample)
l2 = np.linspace(rlmin, rlmax, rsample)

N_phi = 50 # vegas will take many more points
phi1 = np.linspace(0., 2*np.pi, N_phi) # angle between l1 and L
phi2 = np.linspace(0., 2*np.pi, N_phi) # angle between l2 and L

# Reconstruction Ls
Lmin, Lmax = 2, 2000
L = np.linspace(Lmin,Lmax+1,rsample) # or fully sampled: int(Lmax-Lmin+1)


# Load cls
ell = np.arange(0,8000+1,1)

cls_path = './cls'
glclTT = np.loadtxt(f'{cls_path}/glensed_clTT_lmax8000.txt') # we use gradient lensed Cls
lclTT = np.loadtxt(f'{cls_path}/lensed_clTT_lmax8000.txt')
clpp = np.loadtxt(f'{cls_path}/clpp_lmax8000.txt')

# possible configurations: equilateral and folded, squeezed has been implemented but not tested yet.
def get_equil_sep(L):
    return N1B.N1sep_total(L, l1, l2, phi1, phi2, ell, glclTT, lclTT, lclTT, clpp, rlmin, rlmax, 1e5, 'equil', True)
def get_equil_coup(L):
    return N1B.N1coupled_total(L, l1, l2, phi1, phi2, ell, glclTT, lclTT, lclTT, clpp, rlmin, rlmax, 1e5, 'equil', True)

def get_fold_sep(L):
    return N1B.N1sep_total(L, l1, l2, phi1, phi2, ell, glclTT, lclTT, lclTT, clpp, rlmin, rlmax, 1e5, 'fold', True)
def get_fold_sep(L):
    return N1B.N1coupled_total(L, l1, l2, phi1, phi2, ell, glclTT, lclTT, lclTT, clpp, rlmin, rlmax, 1e5, 'fold', True)


data = np.zeros((len(L), 2))
data[:, 0] = np.copy(L)

pool = Pool(ncpus=int(os.getenv('OMP_NUM_THREADS')))
data[:, 1] = np.array(pool.map(get_equil_coup, L)).flatten()

np.savetxt(f'./output/coupN1equil_rlmax{rlmax}Lmax{Lmax}_gCl_lCl_lCl.txt', data)

