import os, sys
import numpy as np
import camb
from camb import model, initialpower
from scipy import integrate
from scipy import special
from tqdm import tqdm

from scipy import optimize
from scipy import interpolate
from scipy.interpolate import UnivariateSpline as spl
from scipy.interpolate import RectBivariateSpline

#Plots
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['mathtext.fontset'] = 'cm'