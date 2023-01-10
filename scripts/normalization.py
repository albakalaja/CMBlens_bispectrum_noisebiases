"""
Last update: February 2022

Authors: Alba Kalaja, Giorgio Orlando

Compute the flat-sky normalization of quadratic estimator.
"""

import numpy as np
from numba import jit, njit, prange
import vegas as vg
from scipy import special, integrate

# load some useful modules
import scripts.polar_coordinates as polar
import scripts.response_functions as rf
import scripts.weight_functions as wf

def get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5):
    """ Calculates the flat-sky normalization of the quadratic estimator, as in eq. (2.13) of https://arxiv.org/abs/2210.16203, with vegas.
    
    Input: - L = array or value of the lensing multipole.
           - l_1 = array or value of l1.
           - phi1 = angle between L and l1.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l_1.
           - nEval = number of evaluations of the integrand in vegas.
              
    Output: Normalization (phi).
    """
    
    def integrand_norm(l_1,phi1):
  
        Lminusl_1 = polar.L1minusl1(L,l_1,phi1)
        f1 = rf.fL1l1_TT(L,l_1,phi1,l_camb,Cl)
        F1 = wf.FL1l1_TT(L,l_1,phi1,l_camb,Cl,expCl)
     
        integrand = l_1*f1*F1/(2.*np.pi)**2 
        
        idx = np.where((l_1 < l1min) | (l_1 > l1max) | (Lminusl_1 < l1min) | (Lminusl_1 > l1max))
        integrand[idx] = 0.
        return integrand
    
    integ = vg.Integrator([[l1min, l1max],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_norm(x[:,0], x[:,1])
    integ(f, nitn=10, neval=nEval)
    result = integ(f, nitn=10, neval=nEval)
    if result.mean != 0:
        return 1./result.mean
    else:
        return 0
    

    
def get_norm_simps(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max):
    """ Calculates the flat-sky normalization of the quadratic estimator, as in eq. (2.13) of https://arxiv.org/abs/2210.16203, with simps integration (less accurate than vegas).
    
    Input: - L = array or value of the lensing multipole.
           - l_1 = array or value of l1.
           - phi1 = angle between L and l1.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l_1.
              
    Output: Normalization (phi).
    """
    def integrand_norm(l_1,phi1):
        Lminusl_1 = polar.L1minusl1(L,l_1,phi1) # L1-l1
        f1 = rf.fL1l1_TT(L,l_1,phi1,l_camb,Cl) # f_TT(l1, L1-l1)
        F1 = wf.FL1l1_TT(L,l_1,phi1,l_camb,Cl,expCl) # F_TT(l1, L1-l1)
     
        integrand = l_1*f1*F1/(2.*np.pi)**2 
        
        idx = np.where((l_1 < l1min) | (l_1 > l1max) | (Lminusl_1 < l1min) | (Lminusl_1 > l1max))
        integrand[idx] = 0.
        return integrand
    
    int_1 = np.zeros(len(phi1))
    for i in range(len(phi1)):
        intgnd = integrand_norm(l_1, phi1[i])
        int_1[i] = integrate.simps(intgnd, x=l_1, even='avg')
    int_ll = integrate.simps(int_1, x=phi1, even='avg')
    result = 1./int_ll

    if not np.isfinite(result):
        result = 0.

    if result < 0.:
        print(L)

    return result
 