"""
Last update: February 2022

Authors: Alba Kalaja, Giorgio Orlando

This module contains all the response functions [f(..)] that are called in the main module for the computation of the noise biases.
"""

import numpy as np
from numba import jit, njit, prange 
import scripts.polar_coordinates as polar

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

def fL1l1_TT(L1, l_1, phi1, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l1, L1-l1), used in the normalization, N0 and N1f.
    """
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    Cl1 = fCl(l_1, l_camb, Cl)
    Cl1L1 = fCl(L1_minusl1, l_camb, Cl)
    
    L1dotl1 = L1*l_1*np.cos(phi1)

    result = Cl1*L1dotl1 + Cl1L1*(L1**2-L1dotl1)
    
    return result


def fl1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(-l1, L2+l1), used in N0.
    """
    L2_plusl1 = polar.L2plusl1(L1, L2, L3, l_1, phi1)
    
    Cl1 = fCl(l_1, l_camb, Cl)
    Cl1L2 = fCl(L2_plusl1, l_camb, Cl)
    
    L2dotl1 = L2*l_1*polar.cos_L2l1(L1, L2, L3, phi1)
    
    result = Cl1L2*(L2**2+L2dotl1) - Cl1*L2dotl1
    
    return result


def fL1l1L2_TT(L1, L2, L3, l_1, phi1, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(L1-l1, L2+l1), used in N0. 
    """
    
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)
    L2_plusl1 = polar.L2plusl1(L1, L2, L3, l_1, phi1)
    Cl1L1 = fCl(L1_minusl1, l_camb, Cl)
    Cl1L2 = fCl(L2_plusl1, l_camb, Cl)
    
    L1dotL3 = L1*L3*polar.cos_L1L3(L1, L2, L3)
    l1dotL3 = l_1*L3*polar.cos_L3l1(L1, L2, L3, phi1)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)

    first_term = l1dotL3-L1dotL3
    second_term = -L2dotL3-l1dotL3
    
    result = Cl1L1*first_term + Cl1L2*second_term
    
    return result


#############################################


def fl2L2_TT(L1, L2, L3, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2, L2-l2), used in N1f.
    """
    L2_minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
    Cl2 = fCl(l_2, l_camb, Cl)
    Cl2L2 = fCl(L2_minusl2, l_camb, Cl)
    
    L2dotl2 = L2*l_2*polar.cos_L2l2(L1, L2, L3, phi2)

    result = Cl2*L2dotl2 + Cl2L2*(L2**2-L2dotl2)
    
    return result


def fL2l2L1_TT(L1, L2, L3, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(L2-l2, L1+l2), used in N1f. 
    """    
    L2minusl_2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)
    L1plusl_2 = polar.L1plusl2(L1, l_2, phi2)
    
    Cl2L2 = fCl(L2minusl_2, l_camb, Cl)
    Cl2L1 = fCl(L1plusl_2, l_camb, Cl)
    
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    L1dotL3 = L1*L3*polar.cos_L1L3(L1, L2, L3)
    
    first_term = l2dotL3-L2dotL3
    second_term = -L1dotL3-l2dotL3
    
    result = Cl2L2*first_term + Cl2L1*second_term
    
    return result


def fl2L1_TT(L1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2, -(L1+l2)), used in N1f.
    """
    L1_plusl2 = polar.L1plusl2(L1, l_2, phi2)
    
    Cl2 = fCl(l_2, l_camb, Cl)
    Cl2L1 = fCl(L1_plusl2, l_camb, Cl)
    
    L1dotl2 = L1*l_2*np.cos(phi2)

    result = -Cl2*L1dotl2 + Cl2L1*(L1**2 + L1dotl2)
    
    return result


def fl1l2_TT(l_1, l_2, phi1, phi2, l_camb, Cl):
    """
    Input: - l_1,l_2 = array or value of l1, l2.
           - phi1,phi2 = angle between L1 and l1, L1 and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l1, l2), used in N1f, N1s, N1t.
    """
    Cl1 = fCl(l_1, l_camb, Cl)
    Cl2 = fCl(l_2, l_camb, Cl)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2) 
    
    first_term = (l_1**2+l1dotl2)
    second_term = (l_2**2+l1dotl2)

    result = Cl1*first_term + Cl2*second_term
    
    return result

def fl1l2_TT2(L, l_1, l_2, phi1, phi2, l_camb, Cl):
    """
    Input: - L = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = unlensed CMB spectrum (CAMB).
           
    Output: f_TT(l1, -l2)
    """
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2) 
    C_l1 = fCl(l_1, l_camb, Cl)
    C_l2 = fCl(l_2, l_camb, Cl)
    
    result = C_l1*(l_1**2-l1dotl2) + C_l2*(-l1dotl2+l_2**2)
    
    return result

def fL1l1l2_TT(L1, l_1, phi1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(L1-l1, -(L1+l2)), used in N1f.
    """
    L1_plusl2 = polar.L1plusl2(L1, l_2, phi2)
    L1_minusl1 = polar.L1minusl1(L1, l_1, phi1)

    Cl1L1 = fCl(L1_minusl1, l_camb, Cl)
    Cl2L1 = fCl(L1_plusl2, l_camb, Cl)

    l1dotL1 = l_1*L1*np.cos(phi1)
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l2dotL1 = l_2*L1*np.cos(phi2)
    
    first_term = l_1**2 - l1dotL1 + l1dotl2 - l2dotL1
    second_term = l_2**2 + l1dotL1 + l1dotl2 + l2dotL1
    
    result = Cl1L1*first_term + Cl2L1*second_term
    
    return result

def fL1l1l2_TT2(L, l_1, l_2, phi1, phi2, l_camb, Cl):
    """
    Input: - L = array or value of the lensing multipole.
           - l_1, l_2 = array or value of l1, l2.
           - phi1, phi2 = angle between L and l1, L and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = unlensed CMB spectrum (CAMB).
           
    Output: f_TT(L1-l1,l2-L1)
    """                
    Lminusl1 = polar.L1minusl1(L, l_1, phi1)
    Lminusl2 = polar.L1minusl2(L, l_2, phi2)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    Ldotl_1 = L*l_1*np.cos(phi1)
    Ldotl_2 = L*l_2*np.cos(phi2)
    C_Ll1 = fCl(Lminusl1, l_camb, Cl)
    C_Ll2 = fCl(Lminusl2, l_camb, Cl)
    
    result = -C_Ll1*(-l_1**2+l1dotl2+Ldotl_1-Ldotl_2)
    result += C_Ll2*(l_2**2-l1dotl2+Ldotl_1-Ldotl_2)
    
    return result

#############################################


def fl1L2_TT2(L1, L2, L3, l_1, phi1, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(l1, L2-l1), used in N1s.
    """
    L2_minusl1 = polar.L2minusl1(L1, L2, L3, l_1, phi1)

    Cl1 = fCl(l_1,l_camb,Cl)
    Cl1L2 = fCl(L2_minusl1,l_camb,Cl)

    L2dotl1 = L2*l_1*polar.cos_L2l1(L1, L2, L3, phi1)
    
    result = Cl1*L2dotl1 + Cl1L2*(L2**2-L2dotl1)
    
    return result


def fl2L1_TT2(L1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(l2, L1-l2), used in N1s, N1t.
    """
    L1_minusl2 = polar.L1minusl2(L1, l_2, phi2)

    Cl2 = fCl(l_2, l_camb, Cl)
    Cl2L1 = fCl(L1_minusl2, l_camb, Cl)    
    
    L1dotl2 = L1*l_2*np.cos(phi2)
    
    result = Cl2*L1dotl2 + Cl2L1*(L1**2-L1dotl2)
    
    return result


def fL1l2L2_TT(L1, L2, L3, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(L1-l2, L2+l2), used in N1s.
    """
    L1minusl_2 = polar.L1minusl2(L1, l_2, phi2)
    L2plusl_2 = polar.L2plusl2(L1, L2, L3, l_2, phi2)
    
    Cl2L1 = fCl(L1minusl_2,l_camb,Cl)
    Cl2L2 = fCl(L2plusl_2,l_camb,Cl)
    
    L1dotL2 = L1*L2*polar.cos_L1L2(L1, L2, L3)
    l2dotL2 = L2*l_2*polar.cos_L2l2(L1, L2, L3, phi2)
    l2dotL1 = L1*l_2*np.cos(phi2)
    
    first_term = L1**2 + L1dotL2 - l2dotL1 - l2dotL2
    second_term = L2**2 + L1dotL2 + l2dotL2 + l2dotL1
    
    result = Cl2L1*first_term + Cl2L2*second_term
    
    return result


def fl2L2_TT2(L1,L2,L3,l_2,phi2,l_camb,Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(l2, -(L2+l2)), used for N1s.
    """
    L2_plusl2 = polar.L2plusl2(L1, L2, L3, l_2, phi2)
    
    Cl2 = fCl(l_2, l_camb, Cl)
    Cl2L2 = fCl(L2_plusl2, l_camb, Cl)
    
    L2dotl2 = L2*l_2*polar.cos_L2l2(L1, L2, L3, phi2)
    
    result = -Cl2*L2dotl2 + Cl2L2*(L2**2+L2dotl2)
    
    return result


def fL2l1l2_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output: f_TT(L2-l1, -(L2+l2)), used for N1s.
    """
    L2_plusl2 = polar.L2plusl2(L1, L2, L3, l_2, phi2)
    L2_minusl1 = polar.L2minusl1(L1, L2, L3, l_1, phi1)

    Cl1L2 = fCl(L2_minusl1, l_camb, Cl)
    Cl2L2 = fCl(L2_plusl2, l_camb, Cl)
    
    l1dotL2 = l_1*L2*polar.cos_L2l1(L1, L2, L3, phi1)
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2) 
    l2dotL2 = l_2*L2*polar.cos_L2l2(L1, L2, L3, phi2)
    
    first_term = (l_1**2 - l1dotL2 + l1dotl2 - l2dotL2)
    second_term = (l_2**2 + l1dotL2 + l1dotl2 + l2dotL2)
    
    result = Cl1L2*first_term + Cl2L2*second_term
    
    return result


#############################################


def fl1L3_TT(L1, L2, L3, l_1, phi1, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output: f_TT(l1, L3-l1), used for N1t.
    """
    L3_minusl1 = polar.L3minusl1(L1, L2, L3, l_1, phi1)
    Cl1 = fCl(l_1, l_camb, Cl)
    Cl1L3 = fCl(L3_minusl1, l_camb, Cl)
    
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    
    result = Cl1*L3dotl1 + Cl1L3*(L3**2-L3dotl1)
    
    return result


def fL1l2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(L1-l2, L3+l2), used in N1t.
    """
    L1minusl_2 = polar.L1minusl2(L1, l_2, phi2)
    L3plusl_2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)
    
    Cl2L1 = fCl(L1minusl_2, l_camb, Cl)
    Cl2L3 = fCl(L3plusl_2, l_camb, Cl)
    
    L1dotL2 = L1*L2*polar.cos_L1L2(L1,L2,L3)
    l2dotL2 = l_2*L2*polar.cos_L2l2(L1, L2, L3, phi2)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    
    first_term = l2dotL2 - L1dotL2
    second_term = -L2dotL3 - l2dotL2
    
    result = Cl2L1*first_term + Cl2L3*second_term
    
    return result


def fl2L3_TT(L1, L2, L3, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(l2, -(L3+l2)), used in N1t.
    """
    L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)
    
    Cl2 = fCl(l_2, l_camb, Cl)
    Cl2L3 = fCl(L3_plusl2, l_camb, Cl)
    
    L3dotl2 = L3*l_2*polar.cos_L3l2(L1, L2, L3, phi2)
    
    result = -Cl2*L3dotl2 + Cl2L3*(L3**2+L3dotl2)
    
    return result


def fL3l1l2_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).

    Output:  f_TT(L3-l1, -(L3+l2)), used in N1t.
    """
    L3_plusl2 = polar.L3plusl2(L1, L2, L3, l_2, phi2)
    L3_minusl1 = polar.L3minusl1(L1, L2, L3, l_1, phi1)

    Cl1L3 = fCl(L3_minusl1, l_camb, Cl)
    Cl2L3 = fCl(L3_plusl2, l_camb, Cl)
    
    l1dotL3 = l_1*L3*polar.cos_L3l1(L1, L2, L3, phi1)
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)
    
    first_term = (l_1**2 - l1dotL3 + l1dotl2 - l2dotL3)
    second_term = (l_2**2 + l1dotL3 + l1dotl2 + l2dotL3)
    
    result = Cl1L3*first_term + Cl2L3*second_term
    
    return result


def fl3L3_TT(L1, L2, L3, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l3, L3-l3), used in N2.
    """
    L3_minusl3 = polar.L3minusl3(L1, L2, L3, l_3, phi3)

    Cl3 = fCl(l_3, l_camb, Cl)
    Cl3L3 = fCl(L3_minusl3, l_camb, Cl)

    L3dotl3 = L3*l_3*polar.cos_L3l3(L1, L2, L3, phi3)

    result = Cl3*L3dotl3 + Cl3L3*(L3**2-L3dotl3)
    
    return result


def fl2l3L2L1_TT(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2-l3-L2, l3-l2-L1), used in N2_1c.
    """
    l2_minusl3minusL2 = polar.l2minusl3minusL2(L1, L2, L3, l_2, phi2, l_3, phi3)
    l3_minusl2minusL1 = polar.l3minusl2minusL1(L1, l_2, phi2, l_3, phi3)

    Cl2l3L2 = fCl(l2_minusl3minusL2, l_camb, Cl)
    Cl3l2L1 = fCl(l3_minusl2minusL1, l_camb, Cl)
    
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)
    l3dotL3 = l_3*L3*polar.cos_L3l3(L1, L2, L3, phi3)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    L1dotL3 = L1*L3*polar.cos_L1L3(L1, L2, L3)

    result = (l2dotL3-l3dotL3-L2dotL3)*Cl2l3L2 + (l3dotL3-l2dotL3-L1dotL3)*Cl3l2L1
    
    return result


def fl2l3L2_TT(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2-l3-L2, L2-l2), used in N2_1c.
    """
    l2_minusl3minusL2 = polar.l2minusl3minusL2(L1, L2, L3, l_2, phi2, l_3, phi3)
    L2_minusl2 = polar.L2minusl2(L1, L2, L3, l_2, phi2)

    Cl2l3L2 = fCl(l2_minusl3minusL2, l_camb, Cl)
    Cl2L2 = fCl(L2_minusl2, l_camb, Cl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l3dotL2 = l_3*L2*polar.cos_L2l3(L1, L2, L3, phi3)

    result = (l_3**2+l3dotL2-l2dotl3)*Cl2l3L2 + (l2dotl3-l3dotL2)*Cl2L2
    
    return result


def fl2l3L1_TT(L1, l_2, phi2, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2, l3-l2-L1), used in N2_1c.
    """
    l3_minusl2minusL1 = polar.l3minusl2minusL1(L1, l_2, phi2, l_3, phi3)

    Cl2 = fCl(l_2, l_camb, Cl)
    Cl3l2L1= fCl(l3_minusl2minusL1, l_camb, Cl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l2dotL1 = l_2*L1*np.cos(phi2)
    l3dotL1 = l_3*L1*np.cos(phi3)

    result = (l2dotl3-l2dotL1)*Cl2 + (l_3**2-2*l3dotL1-l2dotl3+l2dotL1+L1**2)*Cl3l2L1
    
    return result


def fl2L3l3_TT(L1, L2, L3, l_2, phi2, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2, L3-l3), used in N2_1d.
    """
    L3_minusl3 = polar.L3minusl3(L1, L2, L3, l_3, phi3)

    Cl3 = fCl(l_3, l_camb, Cl)
    Cl3l3= fCl(L3_minusl3, l_camb, Cl)
    
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)
    l3dotL3 = l_3*L3*polar.cos_L3l3(L1, L2, L3, phi3)

    result = (l_2**2-l2dotl3+l2dotL3)*Cl3 + (L3**3-2*l3dotL3+l2dotL3-l2dotl3+l_3**2)*Cl3l3
    
    return result


def fL3l3L1_TT(L1, L2, L3, l_3, phi3, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l3-L3, -l3-L1), used in N2_2b.
    """
    L3_minusl3 = polar.L3minusl3(L1, L2, L3, l_3, phi3)
    L1_plusl3 = polar.L1plusl3(L1, l_3, phi3)

    CL1l3 = fCl(L1_plusl3, l_camb, Cl)
    CL3l3= fCl(L3_minusl3, l_camb, Cl)
    
    l3dotL2 = L2*l_3*polar.cos_L2l3(L1, L2, L3, phi3)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    L1dotL2 = L1*L2*polar.cos_L1L2(L1, L2, L3)

    result = (l3dotL2-L2dotL3)*CL3l3 + (-l3dotL2-L1dotL2)*CL1l3
    
    return result


def fl1l2L3L1_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(-l1-l2-L3, l1+l2-L1), used in N2_3a.
    """
    L3_plusl1plusl2 = polar.L3plusl1plusl2(L1, L2, L3, l_1, phi1, l_2, phi2)
    l1_plusl2minusL1 = polar.L1minusl1minusl2(L1, l_1, phi1, l_2, phi2)

    CL3l1l2 = fCl(L3_plusl1plusl2, l_camb, Cl)
    Cl1l2L1= fCl(l1_plusl2minusL1, l_camb, Cl)
    
    l1dotL2 = L2*l_1*polar.cos_L2l1(L1, L2, L3, phi1)
    l2dotL2 = l_2*L2*polar.cos_L2l2(L1, L2, L3, phi2)
    L2dotL3 = L2*L3*polar.cos_L2L3(L1, L2, L3)
    L1dotL2 = L1*L2*polar.cos_L1L2(L1, L2, L3)

    result = (-l1dotL2-l2dotL2-L2dotL3)*CL3l1l2 + (l1dotL2+l2dotL2-L1dotL2)*Cl1l2L1
    
    return result


def fl1l2L3_TT2(L1, L2, L3, l_1, phi1, l_2, phi2, l_camb, Cl):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(-l1-l2, l1+l2+L3), used in N2_3a.
    """
    l1_plusl2 = polar.l1plusl2(l_1, l_2, phi1, phi2)
    L3_plusl1plusl2 = polar.L3plusl1plusl2(L1, L2, L3, l_1, phi1, l_2, phi2)

    Cl1l2 = fCl(l1_plusl2,l_camb,Cl)
    CL3l1l2 = fCl(L3_plusl1plusl2,l_camb,Cl)
    
    l1dotL3 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    l2dotL3 = l_2*L3*polar.cos_L3l2(L1, L2, L3, phi2)

    result = (-l1dotL3-l2dotL3)*Cl1l2 + (l1dotL3+l2dotL3+L3**2)*CL3l1l2
    
    return result


def fl1l2_TT3(l_1, l_2, phi1, phi2, l_camb, Cl):
    """
    Input: - l_1,l_2 = array or value of l1, l2.
           - phi1,phi2 = angle between L1 and l1, L1 and l2.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l1, -l1-l2), used in N2_3a.
    """
    l1_plusl2 = polar.l1plusl2(l_1, l_2, phi1, phi2)

    Cl1 = fCl(l_1, l_camb, Cl)
    Cl1l2 = fCl(l1_plusl2, l_camb, Cl)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2) 

    result = Cl1*(-l1dotl2) + Cl1l2*(l_2**2+l1dotl2)
    
    return result


def fl1l3L3_TT(L1, L2, L3, l_1, phi1, l_3, phi3, l_camb, Cl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(-l1-l3, L3+l1+l3), used in N2_1a.
    """
    l1_plusl3 = polar.l1plusl3(l_1, l_3, phi1, phi3)
    L3_plusl1plusl3 = polar.L3plusl1plusl3(L1, L2, L3, l_1, phi1, l_3, phi3)
    
    Cl1l3 = fCl(l1_plusl3,l_camb,Cl)
    CL3l1l3 = fCl(L3_plusl1plusl3,l_camb,Cl)
    
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    L3dotl3 = L3*l_3*polar.cos_L3l3(L1, L2, L3, phi3)

    result = Cl1l3*(-L3dotl1 - L3dotl3) + CL3l1l3*(L3dotl1 + L3dotl3 + L3**2)
    
    return result


def fl1l3_TT(l_1, phi1, l_3, phi3, l_camb, Cl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l1, -l1-l3), used in N2_1a.
    """
    l1_plusl3 = polar.l1plusl3(l_1, l_3, phi1, phi3)
    
    Cl1 = fCl(l_1, l_camb, Cl)
    Cl1l3 = fCl(l1_plusl3, l_camb, Cl)
    
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)

    result = Cl1*(-l1dotl3) + Cl1l3*(l1dotl3+l_3**2)
    
    return result


def fl2L3l1l3_TT(L1, L2, L3, l_1, phi1, l_2, phi2, l_3, phi3, l_camb, Cl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l2, L3+l1+l3), used in N2_f. 
    """
    L3_plusl1plusl3 = polar.L3plusl1plusl3(L1, L2, L3, l_1, phi1, l_3, phi3)   

    Cl2 = fCl(l_2, l_camb, Cl)
    CL3l1l3 = fCl(L3_plusl1plusl3, l_camb, Cl)
    
    
    L3dotl2 = L3*l_2*polar.cos_L3l2(L1, L2, L3, phi2)
    L3dotl1 = L3*l_1*polar.cos_L3l1(L1, L2, L3, phi1)
    L3dotl3 = L3*l_3*polar.cos_L3l3(L1, L2, L3, phi3)
    
    l1dotl2 = l_1*l_2*polar.cos_l1l2(phi1, phi2)
    l2dotl3 = l_2*l_3*polar.cos_l2l3(phi2, phi3)
    l1dotl3 = l_1*l_3*polar.cos_l1l3(phi1, phi3)

    result = Cl2*(l_2**2 + L3dotl2 + l1dotl2 + l2dotl3) + CL3l1l3*(l_1**2 + 2*l1dotl3 + 2*L3dotl1 + 2*L3dotl3 + L3**2 + l1dotl2 + l2dotl3 + L3dotl2 + l_3**2)
    
    return result


def fl3L1_TT(L1, l_3, phi3, l_camb, Cl):
    """
    Input: - L_i = array or value of the lensing multipoles.
           - l_i = array or value of l_i.
           - phi_i = angle between L1 and l_i.
           - l_camb = array of multipole of the CAMB spectrum.
           - Cl = CAMB spectrum (unlensed or gradient lensed).
              
    Output:  f_TT(l3, -L1-l3), used in N2_c .
    """
    L1_plusl3 = polar.L1plusl3(L1, l_3, phi3)

    Cl3 = fCl(l_3, l_camb, Cl)
    Cl3L1 = fCl(L1_plusl3, l_camb, Cl)
    
    L1dotl3 = L1*l_3*np.cos(phi3)

    result = - Cl3*L1dotl3 + Cl3L1*(L1**2+L1dotl3)
    
    return result

