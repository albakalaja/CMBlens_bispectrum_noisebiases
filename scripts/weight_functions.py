"""
Last update: February 2022

Authors: Alba Kalaja, Giorgio Orlando

This module contains all the response functions [f(..)] that are called in the main module for the computation of the noise biases.
"""

import numpy as np
from numba import jit
import scripts.polar_coordinates as polar # load module with polar coordinates
import scripts.response_functions as rf # load module with response functions

############################################## 
# interpolate CMB-power spectra

@jit(nopython=True) # numba speeds it up significantly
def fCl(l, l_camb, Cl):
    """
    Input: - l = multipole at which you evaluate the function.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum.
              
    Output: function that interpolates the spectra.
    """
    return np.interp(l, l_camb, Cl) 

#############################################

def FL1l1_TT(L1, l_1, phi1, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(l1, L1-l1), used in norm, N0 and N1f.
    """   
    
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    expCl1 = fCl(l_1, l_camb, expCl)
    expCl1L1 = fCl(L1_minusl1, l_camb, expCl)
    
    numerator = rf.fL1l1_TT(L1, l_1, phi1, l_camb, Cl) #f_TT(l1, L1-l1)
    denominator = 2.*expCl1*expCl1L1
    result = numerator/denominator
    
    return result


def Fl1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(-l1, L2+l1) 
    """
    
    L2_plusl1 = polar.L2plusl1(L1, L2, L3, l_1, phi1)
    expCl1 = fCl(l_1, l_camb, expCl)
    expCl1L2 = fCl(L2_plusl1, l_camb, expCl)
    
    numerator = rf.fl1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl) # f_TT(-l1, L2+l1) 
    denominator = 2.*expCl1*expCl1L2
    result = numerator/denominator
    
    return result


def FL1l1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl):    
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(L1-l1, L2+l1), used in N0. 
    """    
    L1_minusl1 = polar.L1minusl1(L1,l_1, phi1)
    L2_plusl1 = polar.L2plusl1(L1,L2,L3,l_1, phi1)

    expCl1L1 = fCl(L1_minusl1, l_camb, expCl)
    expCl1L2 = fCl(L2_plusl1, l_camb, expCl)
    
    numerator = rf.fL1l1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl) # f_TT(L1-l1, L2+l1)
    denominator = 2.*expCl1L1*expCl1L2
    
    result = numerator/denominator
    
    return result


def Fl2L2_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(l2, L2-l2), used in N1f, N2.
    """       
    L2_minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)

    expCl2 = fCl(l_2, l_camb, expCl)
    expCl2L2 = fCl(L2_minusl2,l_camb,expCl)
    
    numerator = rf.fl2L2_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(l2, L2-l2)
    denominator = 2.*expCl2*expCl2L2
    result = numerator/denominator
    
    return result


def FL2l2L1_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl):
    
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(L2-l2, L1+l2), used in N1f.  
    """
    L2minusl_2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
    L1plusl_2 = polar.L1plusl2(L1, l_2, phi2)

    expCl2L2 = fCl(L2minusl_2,l_camb,expCl)
    expCl2L1 = fCl(L1plusl_2,l_camb,expCl)    
    
    numerator = rf.fL2l2L1_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(L2-l2, L1+l2)
    denominator = 2.*expCl2L2*expCl2L1
    result = numerator/denominator
    
    return result


def Fl1L2_TT2(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).

    Output: F_TT(l1, L2-l1), used in N1s.
    """
    
    L2_minusl1 = polar.L2minusl1(L1, L2, L3, l_1, phi1)

    expCl1 = fCl(l_1, l_camb, expCl)
    expCl1L2 = fCl(L2_minusl1, l_camb, expCl)
    
    numerator = rf.fl1L2_TT2(L1, L2, L3, l_1, phi1, l_camb, Cl) # f_TT(l1, L2-l1)
    denominator = 2.*expCl1*expCl1L2
    result = numerator/denominator
    
    return result


def Fl2L1_TT2(L1, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).

    Output: F_TT(l2, L1-l2), used in N1s, N1t.
    """
    
    L1_minusl2 = polar.L1minusl2(L1, l_2, phi2)
    expCl2 = fCl(l_2, l_camb, expCl)
    expCl2L1 = fCl(L1_minusl2, l_camb, expCl)
    
    numerator = rf.fl2L1_TT2(L1, l_2, phi2, l_camb, Cl) # f_TT(l2, L1-l2)
    denominator = 2.*expCl2*expCl2L1
    result = numerator/denominator
    
    return result


def FL1l2L2_TT(L1,L2,L3,l_2, phi2,l_camb,Cl,expCl):    
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).

    Output: F_TT(L1-l2, L2+l2), used in N1s.
    """
    L1minusl_2 = polar.L1minusl2(L1, l_2, phi2)
    L2plusl_2 = polar.L2plusl2(L1, L2, L3, l_2, phi2)

    expCl2L1 = fCl(L1minusl_2, l_camb, expCl)
    expCl2L2 = fCl(L2plusl_2, l_camb, expCl)
    
    numerator = rf.fL1l2L2_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(L1-l2, L2+l2)
    denominator = 2.*expCl2L1*expCl2L2
    result = numerator/denominator
    
    return result


def Fl1L3_TT(L1, L2, L3, l_1, phi1, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).

    Output: F_TT(l1, L3-l1), used in N1t.
    """
    L3_minusl1 = polar.L3minusl1(L1, L2, L3, l_1, phi1)

    expCl1 = fCl(l_1, l_camb, expCl)
    expCl1L3 = fCl(L3_minusl1, l_camb, expCl)
    
    numerator = rf.fl1L3_TT(L1, L2, L3, l_1, phi1, l_camb, Cl) # f_TT(l1, L3-l1)
    denominator = 2.*expCl1*expCl1L3
    result = numerator/denominator
    
    return result


def FL1l2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).

    Output: F_TT(L1-l2, L3+l2), used in N1t.
    """
    L1minusl_2 = polar.L1minusl2(L1, l_2, phi2)
    L3plusl_2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)

    expCl2L1 = fCl(L1minusl_2, l_camb, expCl)
    expCl2L3 = fCl(L3plusl_2, l_camb, expCl)
    
    numerator = rf.fL1l2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(L1-l2, L3+l2)
    denominator = 2.*expCl2L1*expCl2L3
    result = numerator/denominator
    
    return result


def Fl3L3_TT(L1, L2, L3, l_3, phi3, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(l3, L3-l3), used in N2.
    """       
    L3_minusl3 = polar.L3minusl3(L1, L2, L3, l_3, phi3)
    expCl3 = fCl(l_3, l_camb, expCl)
    expCl3L3 = fCl(L3_minusl3, l_camb, expCl)
    
    numerator = rf.fl3L3_TT(L1, L2, L3, l_3, phi3, l_camb, Cl) # f_TT(l3, L3-l3)
    denominator = 2.*expCl3*expCl3L3
    result = numerator/denominator
    
    return result


def Fl2l3L2L1_TT(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output:  F_TT(l2-l3-L2, l3-l2-L1), used in N2_1c.
    """
    l2_minusl3minusL2 = polar.l2minusl3minusL2(L1, L2, L3, l_2, phi2, l_3, phi3)
    l3_minusl2minusL1 = polar.l3minusl2minusL1(L1, l_2, phi2, l_3, phi3)

    expCl2l3L2 = fCl(l2_minusl3minusL2, l_camb, expCl)
    expCl3l2L1 = fCl(l3_minusl2minusL1, l_camb, expCl)
    
    numerator = rf.fl2l3L2L1_TT(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, Cl) # f_TT(l2-l3-L2, l3-l2-L1)
    denominator = 2*expCl2l3L2*expCl3l2L1
    
    result = numerator/denominator

    return result


def FL3l3L1_TT(L1, L2, L3, l_3, phi3, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output:  F_TT(l3-L3, -l3-L1), used in N2_2b.
    """
    L3_minusl3 = polar.L3minusl3(L1, L2, L3, l_3, phi3)
    L1_plusl3 = polar.L1plusl3(L1, l_3, phi3)

    expCL3l3 = fCl(L3_minusl3, l_camb, expCl)
    expCL1l3 = fCl(L1_plusl3, l_camb, expCl)
    
    numerator = rf.fL3l3L1_TT(L1, L2, L3, l_3, phi3, l_camb, Cl) # f_TT(l3-L3,-l3-L1)
    denominator = 2*expCL3l3*expCL1l3 
    
    result = numerator/denominator

    return result


def Fl1l2L3L1_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output:  F_TT(-l1-l2-L3,l1+l2-L1), used in N2_3a.
    """
    L3_plusl1plusl2 = polar.L3plusl1plusl2(L1, L2, L3, l_1, phi1, l_2, phi2)
    l1_plusl2minusL1 = polar.L1minusl1minusl2(L1, l_1, phi1, l_2, phi2)

    expCL3l1l2 = fCl(L3_plusl1plusl2,l_camb,expCl)
    expCl1l2L1= fCl(l1_plusl2minusL1,l_camb,expCl)
    
    numerator = rf.fl1l2L3L1_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl) # f_TT(-l1-l2-L3,l1+l2-L1)
    denominator = 2*expCL3l1l2*expCl1l2L1 
    
    result = numerator/denominator

    return result


def Fl1l2L3_TT2(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output:  F_TT(-l1-l2,l1+l2+L3), used in N2_3a.
    """
    l1_plusl2 = polar.l1plusl2(l_1, l_2, phi1, phi2)
    L3_plusl1plusl2 = polar.L3plusl1plusl2(L1, L2, L3, l_1, phi1, l_2, phi2)

    expCl1l2 = fCl(l1_plusl2,l_camb,expCl)
    expCL3l1l2 = fCl(L3_plusl1plusl2,l_camb,expCl)
    
    numerator = rf.fl1l2L3_TT2(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl) # f_TT(-l1-l2,l1+l2+L3)
    denominator = 2*expCL3l1l2*expCl1l2 
    
    result = numerator/denominator

    return result


def Fl1l3L3_TT(L1, L2, L3, l_1, phi1, l_3, phi3, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(-l1-l3, L3+l1+l3).
    """       
    l1_plusl3 = polar.l1plusl3(l_1, l_3, phi1, phi3)
    L3_plusl1plusl3 = polar.L3plusl1plusl3(L1, L2, L3, l_1, phi1, l_3, phi3)

    expCl1l3 = fCl(l1_plusl3, l_camb, expCl)
    expCL3l1l3 = fCl(L3_plusl1plusl3, l_camb, expCl)
    
    numerator = rf.fl1l3L3_TT(L1, L2, L3, l_1, phi1, l_3, phi3, l_camb, Cl) # f_TT(-l1-l3, L3+l1+l3)
    denominator = 2.*expCl1l3*expCL3l1l3
    result = numerator/denominator
    
    return result


def Fl2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl, expCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
           - expCl = total spectrum (including noise).
              
    Output: F_TT(l2, -(L3+l2)), used in N2_2a.
    """       
    L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)
    
    expCl2 = fCl(l_2, l_camb, expCl)
    expCl2L3 = fCl(L3_plusl2, l_camb, expCl)
    
    numerator = rf.fl2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl) # f_TT(l2, -(L3+l2))
    denominator = 2.*expCl2*expCl2L3
    result = numerator/denominator
    
    return result


###############################################


def HL1l1L3(L1, L2, L3, l_1, phi1, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(L1-l1,l1,-L3), used in N2_1b.
    """       
    
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    L2_plusl1 = polar.L2plusl1(L1, L2, L3, l_1, phi1)
    L3_plusl1 = polar.L3plusl1(L1, L2, L3, l_1, phi1)
    
    Cl1L1 = fCl(L1_minusl1,l_camb,unlCl)
    Cl1L2 = fCl(L2_plusl1,l_camb,unlCl)
    Cl1 = fCl(l_1,l_camb,unlCl)
    Cl1L3 = fCl(L3_plusl1,l_camb,unlCl)
    
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    L2dotl1 = L2*l_1*polar.cos_L2l1(L1, L2, L3, phi1)
    L2dotL1 = L2*L1*polar.cos_L1L2(L1, L2, L3)
    L3dotL2 = L3*L2*polar.cos_L2L3(L1, L2, L3)
    
    factor1 = L3dotl1*(L2dotL1 - L2dotl1)
    factor2 = (L3dotL2 + L3dotl1)*(L2**2 + L2dotl1)
    factor3 = L3dotl1*L2dotl1
    factor4 = (L3dotl1 + L3**2)*(L2dotl1 + L3dotL2)
    
    result = factor1*Cl1L1 - factor2*Cl1L2 + factor3*Cl1 - factor4*Cl1L3
    
    return result


def HL1l1l3(L1, l_1, phi1, l_3, phi3, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(L1-l1,l1,l3), used in N2_h.
    """       
    
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    L1_minusl1minusl3 = polar.L1minusl1minusl3(L1, l_1, phi1, l_3, phi3)
    l1_minusl3 = polar.l1minusl3(l_1, l_3, phi1, phi3)
    
    Cl1L1 = fCl(L1_minusl1,l_camb,unlCl)
    CL1l1l3 = fCl(L1_minusl1minusl3,l_camb,unlCl)
    Cl1 = fCl(l_1,l_camb,unlCl)
    Cl1l3 = fCl(l1_minusl3,l_camb,unlCl)
    
    l3dotL1 = l_3*L1*np.cos(phi3)
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)
    l1dotL1 = l_1*L1*np.cos(phi1)
    
    factor1 = (l3dotL1-l1dotl3)*(L1**2-l1dotL1-l3dotL1+l1dotl3)
    factor2 = (l_3**2+l3dotL1-l1dotl3)*(l_3**2-2*l3dotL1+l1dotl3-l1dotL1+L1**2)
    factor3 = (l1dotl3)*(l1dotL1-l1dotl3)
    factor4 = (l1dotl3-l_3**2)*(l_3**2+l1dotL1-l3dotL1-l1dotl3)
    
    result = factor1*Cl1L1 - factor2*CL1l1l3 + factor3*Cl1 - factor4*Cl1l3
    
    return result


def Hl3L2l2L1(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(l3,L2-l2,-L1), used in N2_i.
    """       

    L1_plusl3 = polar.L1plusl3(L1, l_3, phi3)
    L2_minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
    L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)

    Cl3 = fCl(l_3, l_camb, unlCl)
    Cl3L1 = fCl(L1_plusl3, l_camb, unlCl)
    CL2l2 = fCl(L2_minusl2, l_camb, unlCl)
    CL3l2 = fCl(L3_plusl2, l_camb, unlCl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)
    l3dotL3 = l_3*L3*polar.cos_L3l3(L1, L2, L3, phi3)
    l3dotL2 = l_3*L2*polar.cos_L2l3(L1, L2, L3, phi3)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l2dotL1 = l_2*L1*np.cos(phi2)
    L1dotL3 = L1*L3*polar.cos_L1L3(L1, L2, L3)
    L1dotL2 = L1*L2*polar.cos_L1L2(L1, L2, L3)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    l2dotL2 = l_2*L2*polar.cos_L2l2(L1, L2, L3, phi2)
    
    factor1 = (l3dotL1)*(l_3**2-l2dotl3-l3dotL3)
    factor2 = (L1**2+l3dotL1)*(l_3**2+l3dotL1-l2dotL1-l2dotl3-L1dotL3-l3dotL3)
    factor3 = (L1dotL2-l2dotL1)*(l3dotL2-l2dotl3-l2dotL2+l_2**2-L2dotL3+l2dotL3)
    factor4 = (-L1dotL3-l2dotL1)*(L3**2+l_2**2+2*l2dotL3-l3dotL3-l2dotl3)
    
    result = -factor1*Cl3 + factor2*Cl3L1 - factor3*CL2l2 + factor4*CL3l2
    
    return result


def Hl1L1l2(L1, l_1, phi1, l_2, phi2, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(l1,L1-l1,l2), used in N2_e.
    """       

    l1_minusl2 = polar.l1minusl2(l_1, l_2, phi1, phi2)
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    L1_minusl1minusl2 = polar.L1minusl1minusl2(L1, l_1, phi1, l_2, phi2)

    Cl1 = fCl(l_1, l_camb, unlCl)
    Cl1l2 = fCl(l1_minusl2, l_camb, unlCl)
    CL1l1 = fCl(L1_minusl1, l_camb, unlCl)
    CL1l1l2 = fCl(L1_minusl1minusl2, l_camb, unlCl)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l1dotL1 = l_1*L1*np.cos(phi1)
    l2dotL1 = l_2*L1*np.cos(phi2)
    
    factor1 = l1dotl2*(l1dotL1-l1dotl2)
    factor2 = (l1dotl2-l_2**2)*(l1dotL1-l2dotL1-l1dotl2+l_2**2)
    factor3 = (l2dotL1-l1dotl2)*(L1**2-l1dotL1-l2dotL1+l1dotl2)
    factor4 = (l2dotL1-l1dotl2-l_2**2)*(L1**2-l1dotL1-2*l2dotL1+l1dotl2+l_2**2)
    
    result = factor1*Cl1 - factor2*Cl1l2 + factor3*CL1l1 - factor4*CL1l1l2
    
    return result


def Hl3L1l2(L1, l_3, phi3, l_2, phi2, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(l3,L1+l3,l2), used in N2_e.
    """       
    l2_plusl3 = polar.l2plusl3(l_2, l_3, phi2, phi3)
    L1_plusl3 = polar.L1plusl3(L1, l_3, phi3)
    l2_minusl3minusL1 = polar.l2minusl3minusL1(L1, l_2, phi2, l_3, phi3)

    Cl3 = fCl(l_3, l_camb, unlCl)
    Cl2l3 = fCl(l2_plusl3, l_camb, unlCl)
    CL1l3 = fCl(L1_plusl3, l_camb, unlCl)
    Cl2l3L1 = fCl(l2_minusl3minusL1, l_camb, unlCl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l2dotL1 = l_2*L1*np.cos(phi2)
    
    factor1 = l2dotl3*(-l3dotL1+l2dotl3)
    factor2 = (l_2**2+l2dotl3)*(l_2**2+l2dotl3-l2dotL1-l3dotL1)
    factor3 = (-l2dotl3-l2dotL1)*(L1**2+l3dotL1-l2dotL1-l2dotl3)
    factor4 = (-l2dotl3-l2dotL1+l_2**2)*(L1**2+l3dotL1-2*l2dotL1-l2dotl3+l_2**2)
    
    result = -factor1*Cl3 + factor2*Cl2l3 - factor3*CL1l3 + factor4*Cl2l3L1
    
    return result


def GL1l1l2l3(L1, l_1, phi1, l_3, phi3, l_2, phi2, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: G(L1-l1,l1+l2-L1,l3), used in N2_b.
    """       
    L1_minusl1minusl3 = polar.L1minusl1minusl3(L1, l_1, phi1, l_3, phi3)
    l1_plusl2minusl3minusL1 = polar.l1plusl2minusl3minusL1(L1, l_1, phi1, l_2, phi2, l_3, phi3)

    CL1l1l3 = fCl(L1_minusl1minusl3, l_camb, unlCl)
    Cl1l2l3L1 = fCl(l1_plusl2minusl3minusL1, l_camb, unlCl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)    
    
    factor1 = (l3dotL1-l1dotl3-l_3**2)**2
    factor2 = (l1dotl3+l2dotl3-l_3**2-l3dotL1)**2
    
    result = l2dotl3*(factor1*CL1l1l3 + factor2*Cl1l2l3L1)
    
    return result


def HL1l1L2l2l3(L1, L2, L3, l_1, phi1, l_2, phi2, l_3, phi3, l_camb, unlCl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(L1-l1,L2-l2,l3), used in N2_1a.
    """       
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    L1_minusl1minusl3 = polar.L1minusl1minusl3(L1, l_1, phi1, l_3, phi3)
    L2_minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
    L2_minusl2minusl3 = polar.L2minusl2minusl3(L1, L2, L3, l_2, phi2, l_3, phi3)
    
    
    Cl1L1 = fCl(L1_minusl1, l_camb, unlCl)
    CL1l1l3 = fCl(L1_minusl1minusl3, l_camb, unlCl)
    Cl2L2 = fCl(L2_minusl2, l_camb, unlCl)
    CL2l2l3 = fCl(L2_minusl2minusl3, l_camb, unlCl)
    
    
    L1dotl1 = L1*l_1*np.cos(phi1)
    L1dotl2 = L1*l_2*np.cos(phi2)
    L1dotl3 = L1*l_3*np.cos(phi3)
    
    L1dotL3 = L1*L3*polar.cos_L1L3(L1, L2, L3)
     
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    L3dotl2 = L3*l_2*polar.cos_L3l2(L1, L2, L3, phi2)
    L3dotl3 = L3*l_3*polar.cos_L3l3(L1, L2, L3, phi3)
    
    L2dotl1 = L2*l_1*polar.cos_L2l1(L1, L2, L3, phi1)
    L2dotl2 = L2*l_2*polar.cos_L2l2(L1, L2, L3, phi2)
    L2dotl3 = L2*l_3*polar.cos_L2l3(L1, L2, L3, phi3)
    
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    
    factor1 = (L1dotl3 - l1dotl3)*(-L1dotL3-L1dotl1-L1dotl2-L1dotl3+L3dotl1+l_1**2+l1dotl2+l1dotl3)
    factor2 = - (L1dotl3 - l1dotl3 - l_3**2)*(-L1dotL3-L1dotl1-L1dotl2-L1dotl3+L3dotl1+l_1**2+l1dotl2+2*l1dotl3+L3dotl3+l2dotl3+l_3**2)
    factor3 = (L2dotl3 - l2dotl3)*(-L2dotL3-L2dotl1-L2dotl2-L2dotl3+L3dotl2+l_2**2+l1dotl2+l2dotl3)
    factor4 = - (L2dotl3 - l2dotl3 - l_3**2)*(-L2dotL3-L2dotl1-L2dotl2-L2dotl3+L3dotl2+l_2**2+l1dotl2+2*l2dotl3+l1dotl3+L3dotl3+l_3**2)
    
    result = factor1*Cl1L1 + factor2*CL1l1l3 + factor3*Cl2L2 + factor4*CL2l2l3
    
    return result


def Hl1L3l2l3(L1, L2, L3, l_1, phi1, l_2, phi2, l_3, phi3, l_camb, unlCl):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l2.
           - phi1 = angle between L1 and l1.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: H(l1,L3+l2,-l3), used in N2_2a.
    """       
    L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)
    l1plusl3 = polar.l1plusl3(l_1, l_3, phi1, phi3)
    L3plusl2plusl3 = polar.L3plusl2plusl3(L1, L2, L3, l_2, phi2, l_3, phi3)
    
    Cl1 = fCl(l_1, l_camb, unlCl)
    Cl1l3 = fCl(l1plusl3, l_camb, unlCl)
    CL3l2 = fCl(L3_plusl2, l_camb, unlCl)
    CL3l2l3 = fCl(L3plusl2plusl3, l_camb, unlCl)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    L3dotl2 = L3*l_2*polar.cos_L3l2(L1, L2, L3, phi2)
    L3dotl3 = L3*l_3*polar.cos_L3l3(L1, L2, L3, phi3)
    
    factor1 = - (l1dotl3)*(l_1**2+l1dotl2+l1dotl3+L3dotl1)
    factor2 = (l1dotl3 + l_3**2)*(l_1**2+l1dotl2+2*l1dotl3+L3dotl1+l_3**2+L3dotl3+l2dotl3)
    factor3 = - (l2dotl3 + L3dotl3)*(L3dotl1 + 2*L3dotl2 + L3dotl3 + L3**2 +l_2**2 + l1dotl2 + l2dotl3) 
    factor4 = (l2dotl3 + L3dotl3 + l_3**2)*(L3dotl1 + 2*L3dotl2 + 2*L3dotl3 + L3**2 +l_2**2 + l1dotl2 + 2*l2dotl3 + l_3**2+l1dotl3) 
    
    result = factor1*Cl1 + factor2*Cl1l3 + factor3*CL3l2 + factor4*CL3l2l3
    
    return result


def GL1l2l3(L1, l_2, l_3, phi2, phi3, l_camb, unlCl):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l2.
           - phi1 = angle between L1 and l1.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed).
              
    Output: G(l3,-L1-l3,l2), used in N2_a.
    """       
    L1_plusl2plusl3 = polar.L1plusl2plusl3(L1, l_2, phi2, l_3, phi3)
    l2_minusl3 = polar.l2minusl3(l_2, l_3, phi2, phi3)
    
    Cl3l2 = fCl(l2_minusl3,l_camb, unlCl)
    CL1l2l3 = fCl(L1_plusl2plusl3,l_camb, unlCl)
    
    
    L1dotl2 = L1*l_2*np.cos(phi2)
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    
    factor1 = L1dotl2
    factor2 = (l2dotl3 - l_2**2)**2
    factor3 = (L1dotl2 + l2dotl3 + l_2**2)**2
    
    result = factor1*(factor2*Cl3l2 + factor3*CL1l2l3)
    
    return result


def Gl1L1l2(L1, l_1, l_2, phi1, phi2, l_camb, unlCl):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l2.
           - phi1 = angle between L1 and l1.
           - l_camb = array of multipole of the CAMB spectrum. 
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output: G(l1,L1-l1,l2), used in N2_c.
    """       
    L1_minusl1minusl2 = polar.L1minusl1minusl2(L1, l_1, phi1, l_2, phi2)
    l1_minusl2 = polar.l1minusl2(l_1, l_2, phi1, phi2)
    
    Cl1l2 = fCl(l1_minusl2, l_camb, unlCl)
    CL1l1l2 = fCl(L1_minusl1minusl2, l_camb, unlCl)

    L1dotl2 = L1*l_2*np.cos(phi2)
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    
    factor1 = L1dotl2
    factor2 = (l1dotl2 - l_2**2)**2
    factor3 = (L1dotl2 - l1dotl2 - l_2**2)**2
    
    result = factor1*(factor2*Cl1l2 + factor3*CL1l1l2)
    
    
    return result