
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


import numpy as np
import matplotlib.ticker as ticker

import matplotlib
from matplotlib.lines import lineStyles

violation_total = []

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

from statsmodels.nonparametric.smoothers_lowess import lowess


TRY_NUMBER = [20,50,100,200,500, 1000,2000, 5000,7000, 10000, 13000, 17000, 20000]
ERRORS = [2.890062586436497, 2.41373026456313, 1.4512580905582904, 3.614346282985414, 0.49434555995796703, 4.26127182873454, 0.2552100753730031, 0.3305713537493328, 0.3702475039674126, 6.608940794627171, 0.211741687532871, 0.15022503440102, 0.0411624836739262]

# TRY_NUMBER = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,28,30,31,36,39,41,46,50,55,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
# ERRORS =[ 0.329739847862762, 0.34001286776974976, 0.31229923871775184, 0.046902217918500605, 0.0545879877304768, 0.08789788057225412, 0.08538972207495078, 0.3178942834861023, 0.03862570539321015, 0.14650355982952, 0.1939738525697567, 0.051410356343319606, 0.3502066834371677, 0.309505803902779, 0.12537505170927926, 0.06522679587381917, 0.14138937262763865, 0.13546180600078994, 0.20328459132167767, 0.2914375827365487, 0.17658393296833036, 0.08591153391332525, 0.43075169912302047, 0.17027465312856857, 0.09763170982289714, 0.06312463509707417, 0.08319265862335276, 0.10803841799455376, 0.07366455668444703, 0.20988602474333826, 0.09092024640946447, 0.08719502256974959, 0.4578564837247805, 0.23814503482503183, 0.13871924543625055, 0.35064861993931534, 0.2904105808612551, 0.23000878304627984, 0.47719544220681437, 0.2752126220825252, 0.35947950357136127, 0.37413117434217813, 0.4128353087451594, 0.3035296931757065, 0.3574373694544898, 0.4357609241001217, 0.3370696288754619]


# Applying Lowess to the data
ERRORS = lowess(ERRORS, TRY_NUMBER, frac=.9)[:, 1]  # The frac parameter controls the degree of smoothing
# ERRORS = lowess(ERRORS, TRY_NUMBER, frac=0.3)[:, 1]  # The frac parameter controls the degree of smoothing


fig, ax = plt.subplots(figsize=(6, 2))
plt.plot(np.array(TRY_NUMBER),ERRORS*10/0.02,color="#0B67B2",linewidth=4,alpha=1)

plt.grid(True)
plt.xlabel(r'Number of Models $(N)$')
# plt.xlabel(r'Look-back Window $(W)$')

plt.ylabel(r'MPC Cost')
# ax.set_yscale('log')  # Sets y-axis to log scale
# ax.set_xscale('log')  # Sets y-axis to log scale


ax.yaxis.set_major_locator(ticker.MultipleLocator(300))  # Set major tick interval to every 10

#
# ax.xaxis.set_major_locator(ticker.MultipleLocator(50))  # Set major tick interval to every 10
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))  # Set major tick interval to every 10
# ax.yaxis.set_major_locator(ticker.MultipleLocator(40))  # Set major tick interval to every 10
#
# plt.legend()
plt.tight_layout()

# plt.show()
plt.savefig('n_effect.png', dpi=300, bbox_inches="tight")












