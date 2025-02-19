"""	Nonlinear MPC using dynamic bicycle model with real-time model adaptation.
"""

__author__ = 'Maitham AL-Sunni'
__email__ = 'maitham@cmu.edu'

import time as tm
import numpy as np
import casadi
import _pickle as pickle

import matplotlib
from matplotlib.lines import lineStyles

# matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
matplotlib.use("pgf")  # Uses a LaTeX-compatible backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import subprocess


from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

from apacrace.params import ORCA
from apacrace.models import Dynamic
from apacrace.tracks import ETHZ
from apacrace.mpc.planner import ConstantSpeed
from apacrace.mpc.nmpc import setupNLP
import multiprocessing as mp
from apacrace.mpc.evaluate_models_vectorized import evaluate_models_vectorized
import os
import imageio
import copy

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True

# n_cores = mp.cpu_count()
# print("number of cores is: ",n_cores)
# pool = mp.Pool(processes=1)

# joblib
# pool = Parallel(n_jobs=n_cores)
# print("number of cores: ",n_cores)

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False

#####################################################################
# default settings
SIM_TIME = 36
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

if not TRACK_CONS:
    SUFFIX = 'NOCONS-'
else:
    SUFFIX = ''

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)
print("check1",id(model))

model_run = Dynamic(**params)

#####################################################################
# Model Bank Setup

N_MODELS = 5000  # number of models in the bank
N_AC_STEPS = 10  # Number of steps to accumulate error
smoothing_mu = 20 # moving avg for MUs
smoothing_mu_over_mod = 10 # getting the avg MUs for the best N models
v_factor = .9
mu_init = 1.

class ExponentialSmoother:
    def __init__(self, alpha=0.3):  # Alpha controls smoothness (0.1-0.3 recommended)
        self.alpha = alpha
        self.smooth_value = None

    def update(self, new_value):
        if self.smooth_value is None:
            self.smooth_value = new_value  # Initialize with first value
        else:
            self.smooth_value = self.alpha * new_value + (1 - self.alpha) * self.smooth_value
        return self.smooth_value  # Return smoothed value

smoother = ExponentialSmoother(alpha=0.2)


def find_closest_point(x, y, raceline):
    # Get raceline coordinates
    x_refs = raceline[0]
    y_refs = raceline[1]
    # Calculate distances to all points
    distances = np.sqrt((x_refs - x) ** 2 + (y_refs - y) ** 2)
    # Find index of minimum distance
    idx = np.argmin(distances)
    return idx, x_refs[idx], y_refs[idx], distances[idx]

def update_friction(Df,Dr,curr_time,style='const_decay') :
    if style is 'const_decay' :
        # if curr_time > 6.2 :
        if curr_time > 14.3 :
            Df -= Df/2600.
            Dr -= Dr/2600.
    elif style is 'sudden' :
        if curr_time > 14.3 and curr_time < 14.5:
        # if curr_time > 10 and curr_time < 10.2:
            Df -= Df/13.
            Dr -= Dr/13.
    elif style is 'no_change' :
        return Df, Dr
    return Df, Dr

MODEL_BANK = []
MODEL_PARAMS = []
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]

Drs = []
Dfs = []

Drs_preds = []
Dfs_preds = []

MUs = []
MU_preds = []


# Define variation percentages for each parameter
variation_dict = {
    # 'mass': 0.15,        # 20% variation
    # 'lf': 0.15,          # 10% variation
    # 'lr': 0.15,          # 10% variation
    # 'Cm1': 0.15,         # 15% variation
    # 'Cm2': 0.15,         # 15% variation
    # 'Cr0': 0.15,         # 15% variation
    # 'Cr2': 0.15,         # 15% variation
    'Br': 0.2,          # 15% variation
    'Cr': 0.1,         # 15% variation
    'Dr': 0.5,          # 15% variation
    'Bf': 0.2,          # 15% variation
    'Cf': 0.1,          # 15% variation
    'Df': 0.5,          # 15% variation
}

fval_history = []


# Create model bank with parameter variations
for i in range(N_MODELS):
    param_variation = params.copy()

    for param_name, variation_percentage in variation_dict.items():
        if param_name in param_variation:
            param_variation[param_name] *= (1 + variation_percentage * np.random.randn())

    MODEL_PARAMS.append(param_variation)
    MODEL_BANK.append(Dynamic(**param_variation))

Bfs_pass = np.array([m.Bf for m in MODEL_BANK])
Cfs_pass = np.array([m.Cf for m in MODEL_BANK])
Dfs_pass = np.array([m.Df for m in MODEL_BANK])
Brs_pass = np.array([m.Br for m in MODEL_BANK])
Crs_pass = np.array([m.Cr for m in MODEL_BANK])
Drs_pass = np.array([m.Dr for m in MODEL_BANK])

params_pass = (Bfs_pass, Cfs_pass, Dfs_pass, Brs_pass, Crs_pass, Drs_pass)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

# Initialize error windows as numpy array
error_windows = np.zeros((N_MODELS, N_AC_STEPS))
window_count = 0

# Setup NLP solvers for all models
print("Setting up NLP solvers for all models...")
nlp_bank = []
for i in range(N_MODELS):
    nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R,
                  MODEL_PARAMS[i],
                  MODEL_BANK[i],
                  track, track_cons=TRACK_CONS)
    nlp_bank.append(nlp)
print("NLP setup complete.")

# just begainning

nlp_initial = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model_run, track, track_cons=TRACK_CONS)


#####################################################################
# closed-loop simulation

# initialize
ref_speeds = []

states = np.zeros([n_states, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts

Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])

Ffy_preds = np.zeros([n_steps+1])
Frx_preds = np.zeros([n_steps+1])
Fry_preds = np.zeros([n_steps+1])

hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []

# Arrays to store model performance metrics
model_switches = []  # Store timesteps when model switches occur
model_mses = []     # Store MSE values at switches
chosen_models = []  # Store which model was chosen

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))


media_dir = "LLA"
os.makedirs(media_dir, exist_ok=True)

# dynamic plot
H = .1
W = .05
dims = np.array([[-H/2.,-W/2.],[-H/2.,W/2.],[H/2.,W/2.],[H/2.,-W/2.],[-H/2.,-W/2.]])

fig_track = track.plot(color='k', grid=False)
fig_track.set_dpi(80)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], '#4B0082', label='Trajectory',alpha=0.65)
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=.5, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
LnP, = ax.plot(states[0,0] + dims[:,0]*np.cos(states[2,0]) - dims[:,1]*np.sin(states[2,0])\
		, states[1,0] + dims[:,0]*np.sin(states[2,0]) + dims[:,1]*np.cos(states[2,0]), 'red', alpha=0.8, label='Current pose')
LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=.5, lw=0.5, color='green', label="ground truth")
LnH2, = ax.plot(hstates2[0], hstates2[1], '-b', marker='o', markersize=.5, lw=0.5, color='blue', label="prediction")
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

# Initialize model selection variables
current_model_idx = 0  # Start with nominal model
uprev = np.zeros(n_inputs)  # Initialize uprev


# main simulation loop
for idt in range(n_steps-horizon):
    print("checkiter", id(model))
    x0 = states[:,idt]

    #  "const_decay" or "sudden" or "no_change"
    model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts, "sudden")
    params['Df'], params['Dr'] = model.Df, model.Dr

    # planner based on BayesOpt
    if idt > N_AC_STEPS+1:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx,
                                         curr_mu=MU_pred, scale=v_factor)
    else:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

    ref_speeds.append(v)
    fval_history.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

    if projidx > 656:
        if laps_completed > 0:
            lap_times[laps_completed] = idt * Ts
            print(lap_times)
        else:
            lap_times[laps_completed] = idt * Ts
            print(lap_times)
        laps_completed += 1
        projidx = 0

    start = tm.time()
    # Use pre-computed NLP solver
    # if idt <= N_AC_STEPS:
    #     chosen_model = model
    # else:
    #     print("\n=== Previous Model Parameters ===")
    #     print(f"Model ID: {current_model_idx}")
    #     print(f"Bf: {chosen_model.Bf}")
    #     print(f"Cf: {chosen_model.Cf}")
    #     print(f"Df: {chosen_model.Df}")
    #     print(f"Br: {chosen_model.Br}")
    #     print(f"Cr: {chosen_model.Cr}")
    #     print(f"Dr: {chosen_model.Dr}")
    #     chosen_model = MODEL_BANK[current_model_idx]
    #     print("\n=== New Model Parameters ===")
    #     print(f"Model ID: {current_model_idx}")
    #     print(f"Bf: {chosen_model.Bf}")
    #     print(f"Cf: {chosen_model.Cf}")
    #     print(f"Df: {chosen_model.Df}")
    #     print(f"Br: {chosen_model.Br}")
    #     print(f"Cr: {chosen_model.Cr}")
    #     print(f"Dr: {chosen_model.Dr}")
    #     # nlp = nlp_bank[current_model_idx]

    if idt <= N_AC_STEPS:
        nlp = nlp_initial
    else:
        nlp = nlp_bank[current_model_idx]

    umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

    inputs[:,idt] = umpc[:,0]
    uprev = inputs[:,idt]  # Update uprev for next iteration
    # print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse: {:.5f}".format(
    #     idt,
    #     current_model_idx,
    #     fval,
    #     end - start,
    #     np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
    # ))

    # update current position with numerical integration (exact model)
    x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
    states[:,idt+1] = x_next[:,-1]
    dstates[:,idt+1] = dxdt_next[:,-1]
    Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])
    Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1] = MODEL_BANK[current_model_idx].calc_forces(states[:,idt], inputs[:,idt], return_slip=False)

    Drs.append(model.Dr)
    Dfs.append(model.Df)

    # Drs_preds.append(MODEL_BANK[current_model_idx].Dr)
    # Dfs_preds.append(MODEL_BANK[current_model_idx].Df)

    if idt <= N_AC_STEPS:
        Drs_preds.append(mu_init * params['mass'] * 9.8 * params['lr'] / (params['lf'] + params['lr']))
        Dfs_preds.append(mu_init * params['mass'] * 9.8 * params['lf'] / (params['lf'] + params['lr']))
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))
        MU_preds.append(copy.deepcopy(mu_init))
    else:
        MUs.append(copy.deepcopy((model.Df + model.Dr) / (9.81 * params['mass'])))

        bestKDr = []
        bestKDf = []
        for best_ind in ind_best_KM:
            bestKDr.append(copy.deepcopy(MODEL_BANK[best_ind].Dr))
            bestKDf.append(copy.deepcopy(MODEL_BANK[best_ind].Df))
        Drs_preds.append(np.mean(bestKDr))
        Dfs_preds.append(np.mean(bestKDf))
        MU_pred = copy.deepcopy(np.mean(np.array(Drs_preds)[-smoothing_mu:])  + np.mean(np.array(Dfs_preds)[-smoothing_mu:]) ) / (9.81 * params['mass'])
        MU_preds.append(smoother.update(MU_pred)*.95)

    # In your main loop:
    if idt > 0:
        # Calculate errors for all models at once
        errors = np.mean((evaluate_models_vectorized(MODEL_BANK, N_MODELS, states[:, idt], inputs[:, idt], Ts, params_pass) - states[0:4, idt + 1]) ** 2, axis=1)

        # Update error windows using numpy operations
        error_windows = np.roll(error_windows, -1, axis=1)
        error_windows[:, -1] = errors
        window_count = min(window_count + 1, N_AC_STEPS)

        # Make decision if we have enough data points
        if window_count >= N_AC_STEPS:
            avg_errors = np.mean(error_windows, axis=1)
            new_model_idx = np.argmin(avg_errors)
            ind_best_KM = avg_errors.argsort()[:smoothing_mu_over_mod]

            if new_model_idx != current_model_idx:
                current_model_idx = new_model_idx
                model_switches.append(idt)
                model_mses.append(avg_errors[new_model_idx])
                chosen_models.append(current_model_idx)

    # forward sim to predict over the horizon
    hstates[:,0] = x0
    hstates2[:,0] = x0
    for idh in range(horizon):
        x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
        hstates[:,idh+1] = x_next[:,-1]
        hstates2[:,idh+1] = xmpc[:,idh+1]

    Hs0.append(copy.deepcopy(hstates[0]))
    Hs1.append(copy.deepcopy(hstates[1]))

    Hs0_2.append(copy.deepcopy(hstates2[0]))
    Hs1_2.append(copy.deepcopy(hstates2[1]))

    end = tm.time()
    print("iter: {}, model: {}, cost: {:.5f}, time: {:.2f}, mse*10000: {:.5f}".format(
        idt,
        current_model_idx,
        fval,
        end - start,
        np.mean(error_windows[current_model_idx])*10000 if window_count > 0 else 0.0
    ))

    mean_cost = np.mean(fval_history)
    print(f"Mean cost at iter {idt}: {mean_cost:.3f}")


#####################################################################
# SAVE VIDEO
#
# # **Custom Indigo-Gold Colormap**
# colors = ["#4B0082", "#FFD700"]  # Indigo to Gold
# cmap = mcolors.LinearSegmentedColormap.from_list("indigo_gold", colors)
# norm = mcolors.Normalize(vmin=0, vmax=n_steps-horizon)  # Normalize time range
#
# # **Create Segments for LineCollection**
# segments = np.array([[[states[0, i], states[1, i]], [states[0, i+1], states[1, i+1]]]
#                      for i in range(n_steps-horizon-1)])
#
# lc = LineCollection([], cmap=cmap, norm=norm, linewidth=2)
# ax.add_collection(lc)

def update(idt):

    ax.set_title(f"Frame {idt}")  # Optional: Add frame counter

    # # Get trajectory up to the current frame
    # new_segments = segments[:idt]
    #
    # # Assign colors to segments based on time index
    # new_colors = [cmap(norm(i)) for i in range(idt)]
    #
    # # Update the LineCollection with new segments and colors
    # lc.set_segments(new_segments)
    # lc.set_array(np.arange(idt))  # Apply color mapping


    LnS.set_xdata(states[0, :idt + 1])
    LnS.set_ydata(states[1, :idt + 1])
    #
    # color = cmap(norm(idt))
    # LnS.set_color(color)  # Set color dynamically

    # #
    # # LnR.set_xdata(xref[0,1:])
    # # LnR.set_ydata(xref[1,1:])
    # #
    LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
    LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))


    # #
    LnH.set_xdata(Hs0[idt])
    LnH.set_ydata(Hs1[idt])

    LnH2.set_xdata(Hs0_2[idt])
    LnH2.set_ydata(Hs1_2[idt])

    plt.tight_layout()

# plots

# Plot model switching performance
plt.figure()
plt.plot(np.array(model_switches)*Ts, chosen_models, 'bx-')
plt.xlabel('Time step')
plt.ylabel('Chosen model index')
plt.title('Model Switching')
plt.xlim(0,SIM_TIME)
plt.ylim(0,N_MODELS)
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/Switching.png')

# plot speed
plt.figure()
vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon],color="#E5AE1C",linewidth=4, label='Actual')
# plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
# plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
plt.plot(time[:n_steps-horizon], ref_speeds,color="#0B67B2",linewidth=4, label='Reference')
plt.xlabel(r'Time (s)')
plt.ylabel(r'Speed ($\frac{ \mathrm{m} }{\mathrm{s}}$)')
plt.title('Speeds')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Speeds.png')


# plot acceleration
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel(r'Time (s)')
plt.ylabel('PWM duty cycle [-]')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/Acc.png')


# plot mus
plt.figure()
plt.plot(time[:n_steps-horizon],MUs,color="#E5AE1C",linewidth=4, label=r"Actual")
plt.plot(time[:n_steps-horizon],MU_preds,color="#0B67B2",linewidth=4,  label=r"Predicted")
plt.grid(True)
plt.xlabel(r'Time (s)')
plt.ylabel(r'$\mu$')
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/MUs.png')

# plot steering angle
plt.figure()
plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel('time (s)')
plt.ylabel('Steering (rad)')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/steering.png')


# plot inertial heading
plt.figure()
plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon],color="#0B67B2",linewidth=4)
plt.xlabel('Time (s)')
plt.ylabel('Orientation (rad)')
plt.grid(True)
plt.tight_layout()
plt.savefig(media_dir+'/orientation.png')


plt.figure()
plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="--", label='Actual Df')
plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="--", label='Actual Dr')
plt.plot(time[:n_steps-horizon], Dfs_preds[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="-", label='Predicted Df')
plt.plot(time[:n_steps-horizon], Drs_preds[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="-", label='Predicted Dr')
plt.xlabel('time (s)')
plt.ylabel('mu*N [N]')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(media_dir+'/Ds.png')



# **Set up Matplotlib Animation**
fps = 30
interval = 1000 / fps  # Convert fps to milliseconds

# ani = animation.FuncAnimation(fig_track, update, frames=n_steps-horizon, interval=interval)


# Create a directory to store frames
image_dir = f"{media_dir}/frames"
os.makedirs(image_dir, exist_ok=True)

# Save frames as PNG images
for i in range(n_steps - horizon):
    update(i)  # Render the frame
    fig_track.savefig(f"{image_dir}/frame_{i:04d}.jpg", dpi=80, quality=85)  # Save as JPEG

print("✅ Frames saved, now creating video...")

# FFmpeg command

# Run FFmpeg

video_path = f"{media_dir}/output_video.mp4"

ffmpeg_cmd = f"ffmpeg -framerate {fps} -i {image_dir}/frame_%04d.jpg -c:v h264_videotoolbox -b:v 5000k -pix_fmt yuv420p {video_path}"
subprocess.run(ffmpeg_cmd, shell=True)


print(f"🎥 Smooth video saved as {video_path}")

for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
print(lap_times)