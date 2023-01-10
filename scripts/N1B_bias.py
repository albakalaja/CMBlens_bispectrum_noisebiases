"""
Last update: October 2022

Authors: Alba Kalaja, Giorgio Orlando

Compute N^1_B as in eqs. (4.2)-(4.3) of https://arxiv.org/abs/2210.16203 with all permutations: N^1_B(L1,L2,L3)+N^1_B(L2,L1,L3)+N^1_B(L3,L1,L2)
"""

import numpy as np
from numba import jit, njit, prange
import vegas as vg

# load some useful modules
import scripts.polar_coordinates as polar
import scripts.response_functions as rf
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

##### N1(L1,L2,L3)

def N1f_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient).
           - expCl = total spectrum (including noise).
           - unlCl = unlensed CAMB spectrum.
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L1,L2,L3) separable, without normalization.
    """
    Clpp_L = fCl(L1, l_camb, Cl_pp)
    # We perform the "separable" term integral:
    
    def integrand_sep(l_2,phi2):
        # multipoles
        L2minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
        L1_plusl2 = polar.L1plusl2(L1, l_2, phi2)
        
        # F(..) functions
        F_L2l2 = wf.Fl2L2_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(l2, L2-l2) 
        F_L2L1l2 = wf.FL2l2L1_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(L2-l2, L1+l2)
        f_L1l2 = rf.fl2L1_TT(L1,l_2,phi2,l_camb,Cl) # f_TT(l2,-(L1+l2))
        
        # Cls 
        ClTT_L2l2 = fCl(L2minusl2,l_camb, unlCl)
        
        integrand = l_2*F_L2l2*F_L2L1l2*f_L1l2*ClTT_L2l2/(2.*np.pi)**2 
        
        idx = np.where((l_2 < l1min)|(l_2 > l1max)|(L2minusl2 < l1min)|(L2minusl2 > l1max)|(L1_plusl2 < l1min)|(L1_plusl2 > l1max))
        
        integrand[idx] = 0.
        return integrand
    
    Isep_L= []
    integ = vg.Integrator([[l1min, l1max],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_sep(x[:,0], x[:,1])
    result = integ(f, nitn=10, neval=nEval) # neval = 1e5 should suffice for this type of integral
    Isep_L.append(result.mean)
    Isep_L = np.array(Isep_L)
    
    return 4.*(Clpp_L*Isep_L) 

def N1f_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient).
           - expCl = total spectrum (including noise).
           - unlCl = unlensed CAMB spectrum.
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L1,L2,L3) coupled, without normalization.
    """
    
    def integrand_N1coupled(l_1,l_2,phi1,phi2):
        #multipoles
        L1_minusl1 = polar.L1minusl1(L1,l_1,phi1)
        L2minusl_2 = polar.L2minusl2(L1,L2,L3,l_2,phi2)
        L1_plusl2 = polar.L1plusl2(L1,l_2,phi2)
        l1_plusl2 = polar.l1plusl2(l_1,l_2,phi1,phi2)
        
        # F(..) functions
        F_L1l1 = wf.FL1l1_TT(L1, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L1-l1)
        F_L2l2 = wf.Fl2L2_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(l2, L2-l2) 
        F_L2L1l2 = wf.FL2l2L1_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(L2-l2, L1+l2)
        f_l1l2 = rf.fl1l2_TT(l_1,l_2,phi1,phi2,l_camb,Cl) # f_TT(l1, l2) 
        f_L1l1l2 = rf.fL1l1l2_TT(L1,l_1,phi1,l_2,phi2,l_camb,Cl) # f_TT(L1-l1, -(L1+l2))
        
        # Cls 
        ClTT_L2l2 = fCl(L2minusl_2, l_camb, unlCl)
        Clpp_l1l2 = fCl(l1_plusl2, l_camb, Cl_pp)
        
        integrand = 8.*l_1*l_2*F_L1l1*F_L2l2*F_L2L1l2*f_l1l2*f_L1l1l2*ClTT_L2l2*Clpp_l1l2/(2.*np.pi)**4   
        
        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(l_2 < l1min)|(l_2 > l1max)|(L2minusl_2 < l1min)|(L2minusl_2 > l1max)|(L1_plusl2 < l1min)|(L1_plusl2 > l1max)|(L1_minusl1 < l1min)|(L1_minusl1 > l1max)|np.isnan(l1_plusl2) == True)
        
        integrand[idx] = 0.
        return integrand
    
    N1_L= []
    integ = vg.Integrator([[l1min, l1max],[l1min, l1max],[0, 2*np.pi],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N1coupled(x[:,0],x[:,1],x[:,2],x[:,3])
    result = integ(f, nitn=10, neval=nEval)
    N1_L.append(result.mean)
    N1_L = np.array(N1_L)
    return N1_L


##### N1(L2,L1,L3)

def N1s_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB lensed spectrum.
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L2,L1,L3) separable, without normalization.
    """
    Clpp_L = fCl(L2, l_camb, Cl_pp)
    
    def integrand_sep(l_2,phi2):
        
        # multipoles
        L1minusl_2 = polar.L1minusl2(L1, l_2, phi2) # |L1-l2|
        L2_plusl2 = polar.L2plusl2(L1, L2, L3, l_2, phi2) # |L2+l2|
        
        # F(..) functions
        F_L1l2 = wf.Fl2L1_TT2(L1, l_2, phi2, l_camb, Cl, expCl) # F_TT(l2, L1-l2) 
        F_L1L2l2 = wf.FL1l2L2_TT(L1,L2,L3,l_2, phi2,l_camb,Cl,expCl) # F_TT(L1-l2, L2+l2)
        f_L2l2 = rf.fl2L2_TT2(L1,L2,L3,l_2,phi2,l_camb,Cl) # f_TT(l2, -(L2+l2))
        
        # Cls 
        ClTT_L1l2 = fCl(L1minusl_2, l_camb, unlCl)
        
        integrand = l_2*F_L1l2*F_L1L2l2*f_L2l2*ClTT_L1l2/(2.*np.pi)**2 
        
        idx = np.where((l_2 < l1min)|(l_2 > l1max)|(L1minusl_2 < l1min)|(L1minusl_2 > l1max)|(L2_plusl2 < l1min)|(L2_plusl2 > l1max))
        
        integrand[idx] = 0.
        integrand[np.isnan(integrand)] = 0.0
        integrand[np.isinf(integrand)] = 0.0
        return integrand
    
    Isep_L= []
    integ = vg.Integrator([[l1min, l1max],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_sep(x[:,0], x[:,1])
    result = integ(f, nitn=10, neval=1e5)
    Isep_L.append(result.mean)
    Isep_L = np.array(Isep_L)
    
    return 4.*Clpp_L*Isep_L

def N1s_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB lensed spectrum.
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L2,L1,L3) coupled, without normalization.
    """
    
    def integrand_N1coupled(l_1,l_2,phi1,phi2):
        
        #multipoles
        L2_minusl1 = polar.L2minusl1(L1, L2, L3, l_1, phi1) # |L2-l1|
        L1minusl_2 = polar.L1minusl2(L1, l_2, phi2) # |L1-l2|
        L2_plusl2 = polar.L2plusl2(L1, L2, L3, l_2, phi2) # |L2+l2|
        l1_plusl2 = polar.l1plusl2(l_1, l_2, phi1, phi2) # |l1+l2|
        
        # F(..) functions
        F_L2l1 = wf.Fl1L2_TT2(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L2-l1)
        F_L1l2 = wf.Fl2L1_TT2(L1, l_2, phi2, l_camb, Cl, expCl) # F_TT(l2, L1-l2) 
        F_L1L2l2 = wf.FL1l2L2_TT(L1,L2,L3,l_2, phi2,l_camb,Cl,expCl) # F_TT(L1-l2, L2+l2)
        f_l1l2 = rf.fl1l2_TT(l_1, l_2, phi1, phi2, l_camb, Cl) # f_TT(l1, l2) 
        f_L2l1l2 = rf.fL2l1l2_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl) # f_TT(L2-l1, -(L2+l2))
        
        # Cls 
        ClTT_L1l2 = fCl(L1minusl_2, l_camb, unlCl)
        Clpp_l1l2 = fCl(l1_plusl2, l_camb, Cl_pp)
        
        integrand = 8.*l_1*l_2*F_L2l1*F_L1l2*F_L1L2l2*f_l1l2*f_L2l1l2*ClTT_L1l2*Clpp_l1l2/(2.*np.pi)**4   
        
        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(l_2 < l1min)|(l_2 > l1max)|(L2_minusl1 < l1min)|(L2_minusl1 > l1max)|(L1minusl_2 < l1min)|(L1minusl_2 > l1max)|(L2_plusl2 < l1min)|(L2_plusl2 > l1max)|np.isnan(l1_plusl2) == True)
        
        integrand[idx] = 0.
        integrand[np.isnan(integrand)] = 0.0
        integrand[np.isinf(integrand)] = 0.0

        return integrand
    
    N1_L= []
    integ = vg.Integrator([[l1min, l1max],[l1min, l1max],[0, 2*np.pi],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N1coupled(x[:,0], x[:,1],x[:,2],x[:,3])
    result = integ(f, nitn=10, neval= nEval)
    N1_L.append(result.mean)
    N1_L = np.array(N1_L)
    
    return N1_L

#############################################
# compute unnormalized N^(1)(L3, L1, L2)

def N1t_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB lensed spectrum.
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L3,L1,L2) separable, without normalization.
    """
    
    def integrand_Isep(l_2,phi2):
        # multipoles
        L1minusl_2 = polar.L1minusl2(L1, l_2, phi2) # |L1-l2|
        L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2) # |L3+l2|
        
        # F(..) functions
        F_L1l2 = wf.Fl2L1_TT2(L1, l_2, phi2, l_camb, Cl, expCl) # F_TT(l2, L1-l2) 
        F_L1L2l2 = wf.FL1l2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl) # F_TT(L1-l2, L3+l2)
        f_L3l2 = rf.fl2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(l2, -(L3+l2))
        
        # Cls 
        ClTT_L1l2 = fCl(L1minusl_2,l_camb,unlCl)
        
        integrand = l_2*F_L1l2*F_L1L2l2*f_L3l2*ClTT_L1l2/(2.*np.pi)**2 
        
        idx = np.where((l_2 < l1min)|(l_2 > l1max)|(L1minusl_2 < l1min)|(L1minusl_2 > l1max)|(L3_plusl2 < l1min)|(L3_plusl2 > l1max))
        
        integrand[idx] = 0.
        return integrand
    
    Isep_L= []
    integ = vg.Integrator([[l1min, l1max],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_Isep(x[:,0], x[:,1])
    result = integ(f, nitn=10, neval=nEval)
    Isep_L.append(result.mean)
    Isep_L = np.array(Isep_L)
    
    return 4.*(Clpp_L*Isep_L)
# 
def N1t_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB lensed spectrum.
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1, l2.
    Output: N^1_B(L3,L1,L2) coupled, without normalization.
    """
    
    # Integral over the "coupled" part
    def integrand_N1coupled(l_1,l_2,phi1,phi2):
        
        #multipoles
        L3_minusl1 = polar.L3minusl1(L1, L2, L3, l_1, phi1) # |L3-l1|
        L1minusl_2 = polar.L1minusl2(L1, l_2, phi2) # |L1-l2|
        L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2) # |L3+l2|
        l1_plusl2 = polar.l1plusl2(l_1, l_2, phi1, phi2) # |l1+l2|
        
        # F(..) functions
        F_L3l1 = wf.Fl1L3_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L3-l1)
        F_L1l2 = wf.Fl2L1_TT2(L1, l_2, phi2, l_camb, Cl, expCl) # F_TT(l2, L1-l2)  
        F_L1L3l2 = wf.FL1l2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl) # F_TT(L1-l2, L3+l2)
        f_l1l2 = rf.fl1l2_TT(l_1, l_2, phi1, phi2, l_camb, Cl) # f_TT(l1, l2) 
        f_L3l1l2 = rf.fL3l1l2_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl) # f_TT(L3-l1, -(L3+l2))
        
        # Cls 
        ClTT_L1l2 = fCl(L1minusl_2, l_camb, unlCl)
        Clpp_l1l2 = fCl(l1_plusl2, l_camb, Cl_pp)
        
        integrand = 8.*l_1*l_2*F_L3l1*F_L1l2*F_L1L3l2*f_l1l2*f_L3l1l2*ClTT_L1l2*Clpp_l1l2/(2.*np.pi)**4   
        
        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(l_2 < l1min)|(l_2 > l1max)|(L3_minusl1 < l1min)|(L3_minusl1 > l1max)|(L1minusl_2 < l1min)|(L1minusl_2 > l1max)|(L3_plusl2 < l1min)|(L3_plusl2 > l1max)|np.isnan(l1_plusl2) == True)
        
        integrand[idx] = 0.
        return integrand
    
    N1_L= []
    integ = vg.Integrator([[l1min, l1max],[l1min, l1max],[0, 2*np.pi],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N1coupled(x[:,0], x[:,1],x[:,2],x[:,3])
    result = integ(f, nitn=10, neval=nEval)
    N1_L.append(result.mean)
    N1_L = np.array(N1_L)
    
    return N1_L

# get the configurations

def N1sep_total(L, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval, shape, norm):
    """
    Input: - L = array or value of the lensing multipole.
           - l_1 = array or value of l1.
           - phi1 = angle between L and l1.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed, lensed or gradient lensed for non-perturbative response).
           - expCl = total spectrum (including noise).
           - unlCl = unlensed CAMB spectrum, lensed for non-perturbative.
           - Cl_pp = CAMB lensing potential power spectrum.
           - l1min, l1max = limits of l1.
           - shape = can be 'equil', 'squeez', 'fold'.
           - norm = True is you want the normalized term, otherwise False.
           
    Output: Zeroth-order bias term of the lensing bispectrum (phi).
    """
    if shape == 'equil':
        L1, L2, L3 = L, L, L
        N1 = N1f_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval) # this suffices, the equil shape is symmetric
        if norm == True:
            AL = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return 3*N1*AL*AL
        elif norm == False:
            return 3*N1
    
    elif shape == 'squeez':
        L1, L2, L3 = l1min, L, L
        N1_f = N1f_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_s = N1s_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_t = N1t_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        if norm == True:
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return (N1_f + N1_s + N1_t)*AL2*AL3
        elif norm == False:
            return N1_f+N1_s+N1_t
    
    elif shape == 'fold':
        L1, L2, L3 = L, L/2, L/2
        N1_f = N1f_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_s = N1s_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_t = N1t_sep(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        if norm == True:
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return (N1_f + N1_s + N1_t)*AL2*AL3
        elif norm == False:
            return N1_f+N1_s+N1_t
        
def N1coupled_total(L, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval, shape, norm):
    """
    Input: - L = array or value of the lensing multipole.
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
        N1 = N1f_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval) # this suffices, the equil shape is symmetric
        if norm == True:
            AL = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return 3*N1*AL*AL*AL
        elif norm == False:
            return 3*N1
    
    elif shape == 'squeez':
        L1, L2, L3 = l1min, L, L
        N1_f = N1f_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_s = N1s_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_t = N1t_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L1, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return (N1_f + N1_s + N1_t)*AL1*AL2*AL3
        elif norm == False:
            return N1_f+N1_s+N1_t
    
    elif shape == 'fold':
        L1, L2, L3 = L, L/2, L/2
        N1_f = N1f_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_s = N1s_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        N1_t = N1t_coupled(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, unlCl, Cl_pp, l1min, l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L1, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L2, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L3, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return (N1_f + N1_s + N1_t)*AL1*AL2*AL3
        elif norm == False:
            return N1_f+N1_s+N1_t
        
        
        
        
        
        
        
# check vegas precision
def N1_4d(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, Cl_pp, l1min, l1max, nEval):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB lensed spectrum.
           - expCl = total spectrum (including noise).
           - l1min, l1max = limits of l1, l2.
    Output: First-order bias term of the lensing bispectrum, without normalization.
    """
    
    def integrand_N1(l_1, l_2, phi1, phi2):
        #multipoles
        L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
        L2minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
        L1_plusl2 = polar.L1plusl2(L1, l_2, phi2)
        l1_plusl2 = polar.l1plusl2(l_1,l_2,phi1,phi2)
        # F(..) functions
        F_L1l1 = wf.FL1l1_TT(L1, l_1, phi1, l_camb, Cl, expCl) # F_TT(l1, L1-l1)
        F_L2l2 = wf.Fl2L2_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(l2, L2-l2) 
        F_L2L1l2 = wf.FL2l2L1_TT(L1,L2,L3,l_2,phi2,l_camb,Cl,expCl) # F_TT(L2-l2, L1+l2)
        f_L1l1 = rf.fL1l1_TT(L1,l_1,phi1,l_camb,Cl) # f_TT(l1, L1-l1)
        f_L1l2 = rf.fl2L1_TT(L1,l_2,phi2,l_camb,Cl) # f_TT(l2,-(L1+l2))
        f_L1l1l2 = rf.fL1l1l2_TT(L1,l_1,phi1,l_2,phi2,l_camb,Cl) # f_TT(L1-l1, -(L1+l2))
        f_l1l2 = rf.fl1l2_TT(l_1,l_2,phi1,phi2,l_camb,Cl) # f_TT(l1, l2) 
        
        # Cls 
        Clpp_L = fCl(L1, l_camb, Cl_pp)
        Clpp_l1l2 = fCl(l1_plusl2, l_camb, Cl_pp)
        ClTT_L2l2 = fCl(L2minusl2,l_camb,expCl)
        
        integrand = 4*l_1*l_2*F_L1l1*F_L2l2*F_L2L1l2*(f_L1l1*f_L1l2*Clpp_L + 2*f_L1l1l2*f_l1l2*Clpp_l1l2)*ClTT_L2l2/(2.*np.pi)**4 
        
        idx = np.where((l_1 < l1min)|(l_1 > l1max)|(L1_minusl1 < l1min)|(L1_minusl1 > l1max)|(l_2 < l1min)|(l_2 > l1max)|(L2minusl2 < l1min)|(L2minusl2 > l1max)|(L1_plusl2 < l1min)|(L1_plusl2 > l1max)|np.isnan(l1_plusl2) == True)
        
        integrand[idx] = 0.
        return integrand
    
    N1_L= []
    integ = vg.Integrator([[l1min, l1max],[l1min, l1max],[0, 2*np.pi],[0, 2*np.pi]])
    @vg.batchintegrand
    def f(x):
        return integrand_N1(x[:,0],x[:,1],x[:,2],x[:,3])
    result = integ(f, nitn=10, neval=nEval)
    N1_L.append(result.mean)
    N1_L = np.array(N1_L)
    print(result.summary())
    return N1_L


def N1_4d_total(L, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, Cl_pp, l1min, l1max, nEval, shape, norm):
    """
    Input: - L = array or value of the lensing multipole.
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
        N1 = N1_4d(L1, L2, L3, l_1, l_2, phi1, phi2, l_camb, Cl, expCl, Cl_pp, l1min, l1max, nEval) 
        if norm == True:
            AL = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N1*AL*AL*AL
        elif norm == False:
            return N1
    
    elif shape == 'squeez':
        L1, L2, L3 = l1min, L, L
        N1 = N1_4d(L1,L2,L3,l_1,l_2,phi1,phi2,l_camb,Cl,expCl,Cl_pp,l1min,l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N1*AL1*AL2*AL3
        elif norm == False:
            return N1
    
    elif shape == 'fold':
        L1, L2, L3 = L, L/2, L/2
        N1 = N1_4d(L1,L2,L3,l_1,l_2,phi1,phi2,l_camb,Cl,expCl,Cl_pp,l1min,l1max, nEval)
        if norm == True:
            AL1 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL2 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            AL3 = n.get_norm(L, l_1, phi1, l_camb, Cl, expCl, l1min, l1max, nEval = 1e5)
            return N1*AL1*AL2*AL3
        elif norm == False:
            return N1
        