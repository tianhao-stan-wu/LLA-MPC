
import time as tm
import numpy as np
import casadi
import _pickle as pickle

import matplotlib
from matplotlib.lines import lineStyles

violation_total = []


# matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# matplotlib.use("pgf")  # Uses a LaTeX-compatible backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import subprocess

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True


from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

from apacrace.params import ORCA
from apacrace.models import Dynamic
from apacrace.tracks import ETHZ, ETHZMobil
from apacrace.mpc.planner import ConstantSpeed
from apacrace.mpc.nmpc import setupNLP
import multiprocessing as mp
from apacrace.mpc.evaluate_models_vectorized import evaluate_models_vectorized
import os
import imageio
import copy
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

TRY_NUMBER = [10,20,30,40,50,75,100,150,200,400,500,750, 1000,1500,2000,2500,3000,3500,4000, 5000,6000,7500, 9000,10000]
ERRORS = np.load("THE_BANK_PLOT.npy")


def moving_average(data, window_size):

    if window_size > len(data) or window_size <= 0:
        raise ValueError("Window size must be positive and not greater than data length.")

    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

moving_average(ERRORS,3)

fig, ax = plt.subplots(figsize=(6, 3))
st = 0
end = -1
plt.plot(np.array(TRY_NUMBER)[4:],(moving_average(ERRORS,3)*10/.02)[2:],color="#0B67B2",linewidth=3,alpha=0.7)
plt.grid(True)
plt.xlabel(r'Number of Models ($N$)')
plt.ylabel(r'Prediction Error')
ax.set_yscale('log')  # Sets y-axis to log scale
ax.set_xscale('log')  # Sets y-axis to log scale


# plt.legend()
plt.tight_layout()

# plt.show()
plt.savefig('num_models.png', dpi=300, bbox_inches="tight")
