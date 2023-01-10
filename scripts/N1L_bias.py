"""
Last update: 12 Aug 2021
Authors: Alba Kalaja, Giorgio Orlando.

Compute the first-order bias of the lensing potential power spectrum (flat-sky), without normalization (you can use normalization.py to compute it).

"""

import numpy as np
import camb
from numba import jit, njit, prange
import vegas as vg

# load some useful modules
import scripts.polar_coordinates as polar
import scripts.response_functions as rf
import scripts.weight_functions as wf

@jit(nopython=True)
def fCl(l,l_camb,Cl):
    """
    Input: - l = multipole at which you evaluate the function.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum.
              
    Output: function that interpolates the spectra.
    """
    return np.interp(l,l_camb,Cl) 


def get_N1(L, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = unlensed CMB spectrum (CAMB).
           - expCl = total CMB spectrum (CAMB)
           - l1min, l1max = limits of l1, l2.
    Output: First-order bias term of the lensing power spectrum.
    """
    def integrand_N1(l_1, l_2, phi1, phi2):
            
        l1minusl2 = polar.l1minusl2(l_1, l_2, phi1, phi2) # |l1-l2|
        Lminusl1 = polar.L1minusl1(L,l_1,phi1) # |L-l1|
        Lminusl2 = polar.L1minusl2(L, l_2, phi2) # |L-l2|
        
        Cpp_l1l2 = fCl(l1minusl2, l_camb, Cl_pp)
        
        F_l1L = wf.FL1l1_TT(L, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L1-l1)
        F_l2L = wf.Fl2L1_TT2(L, l_2, phi2, l_camb, Cl, expCl) # F_TT(l2, L1-l2)
        f_l1l2 = rf.fl1l2_TT2(L, l_1, l_2, phi1, phi2, l_camb, Cl) # f_TT(l1, -l2)
        f_l1l2L = rf.fL1l1l2_TT2(L, l_1, l_2, phi1, phi2, l_camb, Cl) # f_TT(L1-l1,l2-L1)
        
        # dl1*l1*dphi1, dl2*l2*dphi2
        integrand = 2.*l_1*l_2*F_l1L*F_l2L*f_l1l2*f_l1l2L*Cpp_l1l2/(2.*np.pi)**4  

        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(l_2 < l1min)|(l_2 > l1max)|(Lminusl1 < l1min)|(Lminusl1 > l1max)|(Lminusl2 < l1min)|(Lminusl2 > l1max)|np.isnan(l1minusl2) == True|(l1minusl2 < l1min)|(l1minusl2 > l1max))

        integrand[idx] = 0.
        integrand[np.isnan(integrand)] = 0.0
        integrand[np.isinf(integrand)] = 0.0

        return integrand
    
    N1_L= []
    integ = vg.Integrator([[l1min, l1max],[l1min, l1max],[0, 2*np.pi],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N1(x[:,0], x[:,1],x[:,2],x[:,3])
    result = integ(f, nitn=10, neval=nEval) 
    N1_L.append(result.mean)
    N1_L = np.array(N1_L)
    return N1_L
