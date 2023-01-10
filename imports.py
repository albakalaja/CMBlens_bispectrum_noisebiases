import os, sys
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import tqdm
from tqdm import tqdm

# for maps
import healpy as hp
# for cmblensplus 
sys.path.append('../cmblensplus_old/wrap')
import basic
import curvedsky as cs


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