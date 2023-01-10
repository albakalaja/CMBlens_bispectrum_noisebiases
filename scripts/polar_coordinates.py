"""
Last update: February 2022

Authors: Alba Kalaja, Giorgio Orlando

This module contains all the angular relations between the momenta and the moduli of various combinations of the momenta in polar coordinates.
"""

import numpy as np
from numba import jit, njit, prange 

############################################### 
# compute cosines (and sines) between the L_i momenta. 
# We choose a frame where vec(L1) is along the positive x-axis in the 2D flat-sky

def cos_L1L3(L1, L2, L3):
    """
    Input: - Li = array or value of the lensing multipole.
              
    Output: cosine of the angle between L1 and L3.
    """
    numerator = L2**2-L3**2-L1**2
    denominator = 2*L1*L3
    return numerator/denominator


def sin_L1L3(L1, L2, L3): 
    """
    Input: - Li = array or value of the lensing multipole.
              
    Output: sine of the angle between L1 and L3.
    """
    l = np.sqrt(2*L1**2*L2**2 + 2*L2**2*L3**2 + 2*L3**2*L1**2 - L1**4 - L2**4 - L3**4)
    denominator = 2*L1*L3
    return l/denominator


def cos_L1L2(L1, L2, L3):
    """
    Input: - Li = array or value of the lensing multipole.
              
    Output: cosine of the angle between L1 and L2.
    """
    numerator = L3**2-L2**2-L1**2
    denominator = 2*L2*L1
    return numerator/denominator


def sin_L1L2(L1, L2, L3):
    """
    Input: - Li = array or value of the lensing multipole.
              
    Output: sine of the angle between L1 and L2.
    """
    l = np.sqrt(2*L3**2*L2**2 + 2*L2**2*L1**2 + 2*L1**2*L3**2 - L3**4 - L2**4 - L1**4)
    denominator = 2*L2*L1
    return -l/denominator


def cos_L2L3(L1, L2, L3):
    """
    Input: - Li = array or value of the lensing multipole.
              
    Output: sine of the angle between L2 and L3.
    """
    return cos_L1L3(L1, L2, L3)*cos_L1L2(L1, L2, L3) + sin_L1L3(L1, L2, L3)*sin_L1L2(L1, L2, L3)


#############################################
# compute cosines between the L_i and l_i momenta. 


def cos_L2l1(L1, L2, L3, phi1):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L2 and l1.
    """
    return cos_L1L2(L1, L2, L3)*np.cos(phi1) + sin_L1L2(L1, L2, L3)*np.sin(phi1)


def cos_L2l2(L1, L2, L3, phi2):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L2 and l2.
    """
    return cos_L1L2(L1, L2, L3)*np.cos(phi2) + sin_L1L2(L1, L2, L3)*np.sin(phi2)


def cos_L2l3(L1, L2, L3, phi3):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L2 and l3.
    """
    return cos_L1L2(L1, L2, L3)*np.cos(phi3) + sin_L1L2(L1, L2, L3)*np.sin(phi3)


def cos_L3l1(L1, L2, L3, phi1):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L3 and l1.
    """
    return cos_L1L3(L1, L2, L3)*np.cos(phi1) + sin_L1L3(L1, L2, L3)*np.sin(phi1)


def cos_L3l2(L1, L2, L3, phi2):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L3 and l2.
    """
    return cos_L1L3(L1, L2, L3)*np.cos(phi2) + sin_L1L3(L1, L2, L3)*np.sin(phi2)


def cos_L3l3(L1,L2,L3,phi3):
    """
    Input: - Li = array or value of the lensing multipole.
           - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between L3 and l3.
    """
    return cos_L1L3(L1, L2, L3)*np.cos(phi3) + sin_L1L3(L1, L2, L3)*np.sin(phi3)


def cos_l1l2(phi1, phi2):
    """
    Input: - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between l1 and l2.
    """
    return np.cos(phi1)*np.cos(phi2) + np.sin(phi1)*np.sin(phi2) 


def cos_l1l3(phi1,phi3):
    """
    Input: - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between l1 and l3.
    """
    return np.cos(phi1)*np.cos(phi3) + np.sin(phi1)*np.sin(phi3)


def cos_l2l3(phi2,phi3):
    """
    Input: - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between l2 and l3.
    """
    return np.cos(phi2)*np.cos(phi3) + np.sin(phi2)*np.sin(phi3) 

def cos_l1p(phi1, theta):
    """
    Input: - phi_i = angle between L1 and l_i.
              
    Output: sine of the angle between l1 and p.
    """
    return np.cos(phi1)*np.cos(theta) + np.sin(phi1)*np.sin(theta) 

#############################################
# compute the moduli of combinations of L_i/l_i momenta


def L1minusl1(L1, l_1, phi1):
    """
    Input: - L1 = array or value of the lensing multipoles
           - l_1 = array or value of l1
           - phi1 = angle between L1 and l1
              
    Output: |L1-l1|.
    """
    l2 = np.sqrt(L1**2 + l_1**2 - 2*L1*l_1*np.cos(phi1))
    l2[np.where(l2 < 0.000001)] = 0.000001 ## Avoid nasty things
    return l2


def L1plusl2(L1, l_2, phi2):
    """
    Input: - L1 = array or value of the lensing multipoles.
           - l_2 = array or value of l2.
           - phi2 = angle between L1 and l2.
              
    Output: |L1+l2|.
    """
    return np.sqrt(L1**2 + l_2**2 + 2*L1*l_2*np.cos(phi2))


def L1minusl2(L1, l_2, phi2):
    """
    Input: - L1 = array or value of the lensing multipoles.
           - l_2 = array or value of l2.
           - phi2 = angle between L1 and l2.

    Output: |L1-l2|.
    """
    return np.sqrt(L1**2 + l_2**2 - 2*L1*l_2*np.cos(phi2))


def L1plusl3(L1, l_3, phi3):
    """
    Input: - L1 = array or value of the lensing multipoles.
           - l_3 = array or value of l3.
           - phi3 = angle between L3 and l3.

    Output: |L1+l3|.
    """
    return np.sqrt(L1**2 + l_3**2 + 2*L1*l_3*np.cos(phi3))


def L1minusl3(L1, l_3, phi3):
    """
    Input: - L1 = array or value of the lensing multipoles.
           - l_3 = array or value of l3.
           - phi3 = angle between L3 and l3.

    Output: |L1-l3|.
    """
    return np.sqrt(L1**2 + l_3**2 - 2*L1*l_3*np.cos(phi3))


def L2minusl1(L1, L2, L3, l_1, phi1):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l1.
           - phi1 = angle between L1 and l1.

    Output: |L2-l1|.
    """
    cphi21 = cos_L2l1(L1,L2,L3,phi1)
    return np.sqrt(L2**2 + l_1**2 - 2*L2*l_1*cphi21)


def L2plusl1(L1, L2, L3, l_1, phi1):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l1.
           - phi1 = angle between L1 and l1.
        
    Output: |L2+l1|.
    """
    cphi21 = cos_L2l1(L1,L2,L3,phi1)
    return np.sqrt(L2**2 + l_1**2 + 2*L2*l_1*cphi21)


def L2plusl2(L1, L2 ,L3, l_2, phi2):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_2 = array or value of l2.
           - phi2 = angle between L1 and l2.
        
    Output: |L2+l2|.
    """
    cphi22 = cos_L2l2(L1,L2,L3,phi2)
    return np.sqrt(L2**2 + l_2**2 + 2*L2*l_2*cphi22)


def L2minusl2(L1, L2, L3, l_2, phi2):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_2 = array or value of l2.
           - phi2 = angle between L1 and l2.
              
    Output: |L2-l2|.
    """
    cphi22 = cos_L2l2(L1,L2,L3,phi2)
    return np.sqrt(L2**2 + l_2**2 - 2*L2*l_2*cphi22)


def L3minusl1(L1, L2, L3, l_1, phi1):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l1.
           - phi1 = angle between L1 and l1.

    Output: |L3-l1|.
    """
    cphi13 = cos_L3l1(L1,L2,L3,phi1)
    return np.sqrt(L3**2 + l_1**2 - 2*L3*l_1*cphi13)


def L3plusl1(L1,L2,L3,l_1,phi1):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_1 = array or value of l1.
           - phi1 = angle between L1 and l1.

    Output: |L3+l1|.
    """
    cphi31 = cos_L3l1(L1,L2,L3,phi1)
    return np.sqrt(L3**2 + l_1**2 + 2*L3*l_1*cphi31)


def L3plusl2(L1, L2, L3, l_2, phi2):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_2 = array or value of l2.
           - phi2 = angle between L1 and l2.

    Output: |L3+l2|.
    """
    cphi23 = cos_L3l2(L1,L2,L3,phi2)
    return np.sqrt(L3**2 + l_2**2 + 2*L3*l_2*cphi23)


def L3minusl3(L1, L2, L3, l_3, phi3):
    """
    Input: - L1,L2,L3 = array or value of the lensing multipoles.
           - l_3 = array or value of l3.
           - phi3 = angle between L3 and l3.

    Output: |L3-l3|.
    """
    cphi33 = cos_L3l3(L1,L2,L3,phi3)
    return np.sqrt(L3**2 + l_3**2 - 2*L3*l_3*cphi33)


def l1plusl2(l_1, l_2, phi1, phi2):
    """
    Input: - l_1, l_2 = array or value of l1,l2.
           - phi1 = angle between L1 and l1.
           - phi2 = angle between L1 and l2.
              
    Output: |l1+l2|.
    """
    angle_term = cos_l1l2(phi1,phi2) 
    
    return np.sqrt(l_1**2 + l_2**2 + 2.*l_1*l_2*angle_term)


def l1minusl2(l_1, l_2, phi1, phi2):
    """
    Input: - l_1, l_2 = array or value of l1,l2.
           - phi1 = angle between L1 and l1.
           - phi2 = angle between L1 and l2.
              
    Output: |l1-l2|
    """
    angle_term = cos_l1l2(phi1,phi2) 
    
    return np.sqrt(l_1**2 + l_2**2 - 2.*l_1*l_2*angle_term)


def l1minusl3(l_1, l_3, phi1, phi3):
    """
    Input: - l_1, l_3 = array or value of l1,l3.
           - phi1 = angle between L1 and l1.
           - phi3 = angle between L1 and l3.
              
    Output: |l1-l3|.
    """
    angle_term = cos_l1l3(phi1,phi3)
    
    return np.sqrt(l_1**2 + l_3**2 - 2.*l_1*l_3*angle_term)


def l1plusl3(l_1, l_3, phi1, phi3):
    """
    Input: - l_1, l_3 = array or value of l1,l3.
           - phi1 = angle between L1 and l1.
           - phi3 = angle between L1 and l3.
              
    Output: |l1+l3|.
    """
    angle_term = cos_l1l3(phi1,phi3)
    
    return np.sqrt(l_1**2 + l_3**2 + 2.*l_1*l_3*angle_term)


def l2plusl3(l_2, l_3, phi2, phi3):
    """
    Input: - l_1, l_2 = array or value of l1,l2.
           - phi1 = angle between L1 and l1.
           - phi2 = angle between L1 and l2.
              
    Output: |l2+l3|
    """
    angle_term = cos_l2l3(phi2, phi3)
    
    return np.sqrt(l_2**2 + l_3**2 + 2.*l_2*l_3*angle_term)

def l2minusl3(l_2, l_3, phi2, phi3):
    """
    Input: - l_1, l_2 = array or value of l1,l2.
           - phi1 = angle between L1 and l1.
           - phi2 = angle between L1 and l2.
              
    Output: |l2-l3|
    """
    angle_term = cos_l2l3(phi2, phi3)
    
    return np.sqrt(l_2**2 + l_3**2 - 2.*l_2*l_3*angle_term)

def l2minusl3minusL2(L1, L2, L3, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |l2-l3-L2|.
    """
    l2dotL2 = l_2*L2*cos_L2l2(L1, L2, L3, phi2)
    l3dotL2 = l_3*L2*cos_L2l3(L1, L2, L3, phi3)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    
    return np.sqrt(L2**2 + l_2**2 + l_3**2 - 2*l2dotL2 + 2*l3dotL2 - 2*l2dotl3)


def L2minusl2minusl3(L1, L2, L3, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L2-l2-l3|.
    """
    l2dotL2 = l_2*L2*cos_L2l2(L1, L2, L3, phi2)
    l3dotL2 = l_3*L2*cos_L2l3(L1, L2, L3, phi3)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    
    return np.sqrt(L2**2 + l_2**2 + l_3**2 - 2*l2dotL2 - 2*l3dotL2 + 2*l2dotl3)


def l2minusl3minusL1(L1, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |l2-l3-L1|.
    """
    l2dotL1 = l_2*L1*np.cos(phi2)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    
    return np.sqrt(L1**2 + l_2**2 + l_3**2 - 2*l2dotl3 - 2*l2dotL1 + 2*l3dotL1)

def l3minusl2minusL1(L1, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |l3-l2-L1|.
    """
    l2dotL1 = l_2*L1*np.cos(phi2)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    
    return np.sqrt(L1**2 + l_2**2 + l_3**2 + 2*l2dotL1 - 2*l3dotL1 - 2*l2dotl3)


def L1minusl1minusl3(L1, l_1, phi1, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L1-l1-l3|.
    """
    l3dotL1 = l_3*L1*np.cos(phi3)
    l1dotl3 = l_1*l_3*cos_l1l3(phi1, phi3)
    l1dotL1 = l_1*L1*np.cos(phi1)
    
    return np.sqrt(L1**2 + l_1**2 + l_3**2 - 2*l1dotL1 - 2*l3dotL1 + 2*l1dotl3)


def l3minusl2minusL3(L1, L2, L3, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |l3-l2-L3|.
    """
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    l2dotL3 = l_2*L3*cos_L3l2(L1, L2, L3, phi2)
    l3dotL3 = l_3*L3*cos_L3l3(L1, L2, L3, phi3)
    
    return np.sqrt(L3**2 + l_2**2 + l_3**2 - 2*l2dotl3 - 2*l3dotL3 + 2*l2dotL3)


def L1minusl1minusl2(L1, l_1, phi1, l_2, phi2):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L1-l1-l2|.
    """
    l1dotL1 = l_1*L1*np.cos(phi1)
    l1dotl2 = l_1*l_2*cos_l1l2(phi1, phi2)
    l2dotL1 = l_2*L1*np.cos(phi2)
    
    return np.sqrt(L1**2 + l_1**2 + l_2**2 - 2*l1dotL1 - 2*l2dotL1 + 2*l1dotl2)


def L3plusl1plusl2(L1, L2, L3, l_1, phi1, l_2, phi2):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L3+l1+l2|.
    """
    l1dotL3 = l_1*L3*cos_L3l1(L1, L2, L3, phi1)
    l1dotl2 = l_1*l_2*cos_l1l2(phi1, phi2)
    l2dotL3 = l_2*L3*cos_L3l2(L1, L2, L3, phi2)
    
    return np.sqrt(L3**2 + l_1**2 + l_2**2 + 2*l1dotL3 + 2*l2dotL3 + 2*l1dotl2)


def L3plusl1plusl3(L1, L2, L3, l_1, phi1, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L3+l1+l3|.
    """
    l1dotL3 = l_1*L3*cos_L3l1(L1, L2, L3, phi1)
    l1dotl3 = l_1*l_3*cos_l1l3(phi1,phi3)
    l3dotL3 = L3*l_3*cos_L3l3(L1,L2,L3,phi3)
    
    return np.sqrt(L3**2 + l_1**2 + l_3**2 + 2*l1dotL3 + 2*l1dotl3 + 2*l3dotL3)


def L3plusl2plusl3(L1, L2, L3, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           
    Output: |L3+l2+l3|.
    """
    l2dotL3 = L3*l_2*cos_L3l2(L1,L2,L3,phi2)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2,phi3)
    l3dotL3 = L3*l_3*cos_L3l3(L1,L2,L3,phi3)
    
    return np.sqrt(L3**2 + l_2**2 + l_3**2 + 2*l2dotL3 + 2*l2dotl3 + 2*l3dotL3)


def L1plusl1plusl2(L1, l_1, phi1, l_2, phi2):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           
    Output: |L1+l1+l2|.
    """
    l1dotL1 = l_1*L1*np.cos(phi1)
    l2dotL1 = l_2*L1*np.cos(phi2)
    l1dotl2 = l_1*l_2*cos_l1l2(phi1, phi2)
    
    return np.sqrt(L1**2 + l_1**2 + l_2**2 + 2*l1dotL1 + 2*l2dotL1 + 2*l1dotl2)

def L1plusl2plusl3(L1, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.
           
    Output: |L1+l2+l3|.
    """
    l2dotL1 = l_2*L1*np.cos(phi2)
    l3dotL1 = l_3*L1*np.cos(phi3)
    l2dotl3 = l_2*l_3*cos_l1l2(phi2, phi3)
    
    return np.sqrt(L1**2 + l_2**2 + l_3**2 + 2*l2dotL1 + 2*l3dotL1 + 2*l2dotl3)


def l1plusl2minusl3minusL1(L1, l_1, phi1, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |l1+l2-l3-L1|.
    """
    l1dotl2 = l_1*l_2*cos_l1l2(phi1, phi2)
    l1dotL1 = l_1*L1*np.cos(phi1)
    l1dotl3 = l_1*l_3*cos_l1l3(phi1, phi3)    
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    l2dotL1 = l_2*L1*np.cos(phi2)
    l3dotL1 = l_3*L1*np.cos(phi3)

    return np.sqrt(l_1**2 + l_2**2 + l_3**2 + L1**2 + 2*l1dotl2 - 2*l1dotl3 - 2*l1dotL1 - 2*l2dotl3 - 2*l2dotL1 + 2*l3dotL1)


def L3plusl1plusl2plusl3(L1, L2, L3, l_1, phi1, l_2, phi2, l_3, phi3):
    """
    Input: - Li = array or value of the lensing multipoles.
           - l_i = array or value of li.
           - phi_i = angle between L1 and li.

    Output: |L3+l1+l2+l3|.
    """
    l1dotL3 = l_1*L3*cos_L3l1(L1, L2, L3, phi1)
    l1dotl3 = l_1*l_3*cos_l1l3(phi1,phi3)
    l3dotL3 = L3*l_3*cos_L3l3(L1,L2,L3,phi3)
    l2dotL3 = l_2*L3*cos_L3l2(L1, L2, L3, phi2)
    l1dotl2 = l_1*l_2*cos_l1l2(phi1, phi2)
    l2dotl3 = l_2*l_3*cos_l2l3(phi2, phi3)
    
    return np.sqrt(L3**2 + l_1**2 + l_2**2 + l_3**2 + 2*l1dotL3 + 2*l2dotL3 + 2*l3dotL3 + 2*l1dotl3 + 2*l1dotl2 + 2*l2dotl3) 
        

def l1minusp(l_1, p, phi1, theta):
    
    angle_term = cos_l1p(phi1,theta) 
    
    return np.sqrt(l_1**2 + p**2 - 2.*l_1*p*angle_term)
