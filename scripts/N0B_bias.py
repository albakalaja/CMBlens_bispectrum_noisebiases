"""
Last update: October 2022

Authors: Alba Kalaja, Giorgio Orlando

Compute N^0_B as in eq. (3.12) of https://arxiv.org/abs/2210.16203
"""

import numpy as np
from numba import jit, njit, prange
import vegas as vg

# load some useful modules
import scripts.polar_coordinates as polar
import scripts.weight_functions as wf
import scripts.normalization as n

@jit(nopython=True)
def fCl(l, l_camb, Cl):
    """
    Input: - l = multipole at which you evaluate the function.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum.
              
    Output: function that interpolates the spectra.
    """
    return np.interp(l, l_camb, Cl) 

def N0_un(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl, unlCl, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1 = array or value of l1.
           - phi1 = angle between L and l1.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed, lensed or gradient lensed for non-perturbative response).
           - expCl = total spectrum (including noise).
           - unlCl = unlensed CAMB spectrum, lensed for non-perturbative.
           - l1min, l1max = limits of l1.
           - nEval = parameter that fixes the # of evaluations of the MC integral, 1e5 should be fine.
           
    Output: Unnormalized zeroth-order bias term of the lensing bispectrum for a generic shape.
    """
    def integrand_N0(l_1, phi1):
        L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
        L2_plusl1 = polar.L2plusl1(L1, L2, L3, l_1, phi1)
        # F(..) functions
        F_L1l1 = wf.FL1l1_TT(L1, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L1-l1)
        F_L2l1 = wf.Fl1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl) # F_TT(-l1, L2+l1)
        F_L1L2l1 = wf.FL1l1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl) # F_TT(L1-l1 L2+l1)
        # Cls
        expC_l1 = fCl(l_1, l_camb, unlCl)
        expC_L1l1 = fCl(L1_minusl1, l_camb, unlCl)
        expC_L2l1 = fCl(L2_plusl1, l_camb, unlCl)

        integrand = 8.*l_1*F_L1l1*F_L2l1*F_L1L2l1*expC_l1*expC_L1l1*expC_L2l1/(2.*np.pi)**2

        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(L1_minusl1 < l1min)|(L1_minusl1 > l1max)|(L2_plusl1 < l1min)|(L2_plusl1 > l1max)|np.isnan(L2_plusl1) == True)
        
        integrand[idx] = 0.
        integrand[np.isnan(integrand)] = 0.0
        integrand[np.isinf(integrand)] = 0.0
        return integrand
    
    N0_L= []
    integ = vg.Integrator([[l1min, l1max],[0.0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N0(x[:,0], x[:,1])
    result = integ(f, nitn=10, neval=nEval)
    N0_L.append(result.mean)
    
    return np.array(N0_L)

def N0_total(L, l_1, phi1, l_camb, Cl, expCl, unlCl, l1min, l1max, nEval, shape, norm):
    """
    Input: - L = array or value of the lensing multipole.
           - L_shape = array or value of the lensing multipole depending on the shape.
           - l_1 = array or value of l1.
           - phi1 = angle between L and l1.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1.
           - shape = can be 'equil', 'squeez', 'fold'.
           - norm = True is you want the normalized term, otherwise False.
           
    Output: Zeroth-order bias term of the lensing bispectrum (phi).
    """
    if shape == 'equil':
        L1, L2, L3 = L, L, L
        N0 = N0_un(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl, unlCl, l1min, l1max, nEval)  
        if norm == True:
            AL = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N0*AL*AL*AL
        elif norm == False:
            return N0
    
    elif shape == 'squeez':
        L1, L2, L3 = l1min, L, L
        N0 = N0_un(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl, unlCl, l1min, l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L1, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N0*AL1*AL2*AL3
        elif norm == False:
            return N0
    
    elif shape == 'fold':
        L1, L2, L3 = L, int(L/2), int(L/2)
        N0 = N0_un(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl, unlCl, l1min, l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L1, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N0*AL1*AL2*AL3
        elif norm == False:
            return N0