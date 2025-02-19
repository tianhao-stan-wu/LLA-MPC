"""	Nonlinear MPC using dynamic bicycle model with real-time model adaptation.
"""

def compute_mu_pred(Ffy_preds, Fry_preds, Nf, Nr, idt, window_size=10):
    """
    Computes the friction coefficient μ_pred using a rolling max over a window
    starting from the current index `idt` and going backwards.

    Parameters:
    - Ffy_preds: np.array, predicted front lateral force values
    - Fry_preds: np.array, predicted rear lateral force values
    - Nf, Nr: float, normal forces at the front and rear
    - idt: int, current index where the window starts
    - window_size: int, number of points in the rolling window (default=50)

    Returns:
    - MU_pred: float, estimated friction coefficient
    """
    # Ensure we don't go out of bounds
    start_idx = max(0, idt - window_size + 1)
    end_idx = idt + 2  # Include the current index `idt`

    # Compute max values within the window
    max_Ffy = np.max(np.abs(Ffy_preds[start_idx:end_idx]))
    max_Fry = np.max(np.abs(Fry_preds[start_idx:end_idx]))

    # Compute predicted friction coefficient
    MU_pred = (max_Ffy + max_Fry) / (Nf + Nr)

    return MU_pred

def find_closest_point(x, y, raceline):
	"""
    Find closest point on raceline to given (x,y) point

    Args:
        x, y: Query point coordinates
        raceline: Track raceline [x_refs, y_refs]

    Returns:
        idx: Index of closest point
        x_closest, y_closest: Coordinates of closest point
        distance: Distance to closest point
    """
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
		if curr_time > 6.2 :
			Df -= Df/2200.
			Dr -= Dr/2200.
	elif style is 'sudden' :
		if curr_time > 14.3 and curr_time < 14.5:
		# if curr_time > 10 and curr_time < 10.2:
			Df -= Df/22.
			Dr -= Dr/22.
	return Df, Dr

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import time as tm
import numpy as np
import casadi
import _pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed

from apacrace.params import ORCA
from apacrace.models import Dynamic
from apacrace.tracks import ETHZ
from apacrace.mpc.planner import ConstantSpeed
from apacrace.mpc.nmpc import setupNLP
import multiprocessing as mp
from apacrace.mpc.evaluate_models_vectorized import evaluate_models_vectorized, evaluate_batch

n_cores = mp.cpu_count()
pool = mp.Pool(processes=8)

# joblib
# pool = Parallel(n_jobs=n_cores)
# print("number of cores: ",n_cores)

#####################################################################
# CHANGE THIS

SAVE_RESULTS = False
TRACK_CONS = False

#####################################################################
# default settings

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

changing_params = ORCA(control='pwm')
changing_run_model = Dynamic(**changing_params)
print("check1",id(model))

#####################################################################
# Model Bank Setup

N_MODELS = 3000  # number of models in the bank
N_AC_STEPS = 10  # Number of steps to accumulate error

def evaluate_model(model, current_state, input_val, Ts):
    x_next, _ = model.sim_continuous(current_state, input_val.reshape(-1,1), [0, Ts])
    return x_next[:,-1]


#joblib
# def evaluate_models_vectorized(models, current_state, input_val, Ts, pool):
#     batch_size = len(models) // n_cores
#     batches = []
#     for i in range(0, len(models), batch_size):
#         batch_models = models[i:i + batch_size]
#         batches.append((batch_models, current_state, input_val, Ts))
#
#     results = pool(delayed(evaluate_batch)(batch) for batch in batches)
#     return np.vstack(results)
#
# def evaluate_batch(args):
#     models, current_state, input_val, Ts = args
#     predictions = np.empty((len(models), len(current_state)))
#     for i, model in enumerate(models):
#         x_next, _ = model.sim_continuous(current_state, input_val.reshape(-1, 1), [0, Ts])
#         predictions[i] = x_next[:, -1]
#     return predictions


MODEL_BANK = []
MODEL_PARAMS = []
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]

Drs = []
Dfs = []

Brs_preds = []
Bfs_preds = []

Crs_preds = []
Cfs_preds = []

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

# MODEL_PARAMS.append(params)
# MODEL_BANK.append(Dynamic(**params))

# Create model bank with parameter variations
for i in range(N_MODELS):
    param_variation = params.copy()

    for param_name, variation_percentage in variation_dict.items():
        if param_name in param_variation:
            param_variation[param_name] *= (1 + variation_percentage * np.random.randn())

    MODEL_PARAMS.append(param_variation)
    MODEL_BANK.append(Dynamic(**param_variation))

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
SIM_TIME = 36

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

# just begainning
nlp_initial = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, model, track, track_cons=TRACK_CONS)

#####################################################################
# closed-loop simulation

# initialize
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

# dynamic plot
fig = track.plot(color='k', grid=False)
plt.plot(track.x_raceline, track.y_raceline, '--k', alpha=0.5, lw=0.5)
ax = plt.gca()
LnS, = ax.plot(states[0,0], states[1,0], 'r', alpha=0.8)
# LnR, = ax.plot(states[0,0], states[1,0], '-b', marker='o', markersize=1, lw=0.5, label="reference")
xyproj, _ = track.project(x=x_init[0], y=x_init[1], raceline=track.raceline)
# LnP, = ax.plot(xyproj[0], xyproj[1], 'g', marker='o', alpha=0.5, markersize=5, label="current position")
# LnH, = ax.plot(hstates[0], hstates[1], '-g', marker='o', markersize=1, lw=0.5, label="ground truth")
# LnH2, = ax.plot(hstates2[0], hstates2[1], '-r', marker='o', markersize=1, lw=0.5, label="prediction")

plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure()
plt.grid(True)
ax2 = plt.gca()
LnFfy, = ax2.plot(0, 0, label='Ffy')
# LnFrx, = ax2.plot(0, 0, label='Frx')
LnFry, = ax2.plot(0, 0, label='Fry')

LnFfy_preds, = ax2.plot(0, 0, label='Ffy_preds')
# LnFrx_preds, = ax2.plot(0, 0, label='Frx_preds')
LnFry_preds, = ax2.plot(0, 0, label='Fry_preds')

plt.xlim([0, SIM_TIME])
plt.ylim([-params['mass']*9.81, params['mass']*9.81])
plt.xlabel('time [s]')
plt.ylabel('force [N]')
plt.legend()
plt.ion()
plt.show()

# Initialize model selection variables
uprev = np.zeros(n_inputs)  # Initialize uprev

# main simulation loop
for idt in range(n_steps-horizon):
    print("checkiter", id(model))
    x0 = states[:,idt]

    # if idt > 310:
    #     model.Df -= model.Df / 2200.
    #     model.Dr -= model.Dr / 2200.

    model.Df, model.Dr = update_friction(model.Df, model.Dr, idt * Ts, "const_decay")
    params['Df'], params['Dr'] = model.Df, model.Dr

    # planner based on BayesOpt
    if idt > 30:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx,
                                         curr_mu=MU_pred)
    else:
        xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx)

    fval_history.append(find_closest_point(x0[0], x0[1], track.raceline)[-1])

    if projidx > 656:
        if laps_completed > 0:
            lap_times[laps_completed] = idt * Ts  # - lap_times[laps_completed - 1]
            print(lap_times)
        else:
            lap_times[laps_completed] = idt * Ts
            print(lap_times)
        laps_completed += 1
        projidx = 0

    start = tm.time()
    # Use pre-computed NLP solver
    if idt <= N_AC_STEPS:
        nlp = nlp_initial
    else:
        nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, changing_params, changing_run_model, track, track_cons=TRACK_CONS)

    umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev)

    # end = tm.time()
    inputs[:,idt] = umpc[:,0]
    uprev = inputs[:,idt]  # Update uprev for next iteration


    # update current position with numerical integration (exact model)
    x_next, dxdt_next = model.sim_continuous(states[:,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
    states[:,idt+1] = x_next[:,-1] #*(np.ran)
    dstates[:,idt+1] = dxdt_next[:,-1]
    Ffy[idt+1], Frx[idt+1], Fry[idt+1] = model.calc_forces(states[:,idt], inputs[:,idt])
    Ffy_preds[idt+1], Frx_preds[idt+1], Fry_preds[idt+1], alphaf, alphar = changing_run_model.calc_forces(states[:,idt], inputs[:,idt], return_slip=True)

    Drs.append(model.Dr)
    Dfs.append(model.Df)

    if idt > 12:
        MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))

        bestKBr = []
        bestKBf = []


        bestKCr = []
        bestKCf = []

        bestKDr = []
        bestKDf = []

        for best_ind in ind_best_KM:
            bestKBr.append(MODEL_BANK[best_ind].Br)
            bestKBf.append(MODEL_BANK[best_ind].Bf)

            bestKCr.append(MODEL_BANK[best_ind].Cr)
            bestKCf.append(MODEL_BANK[best_ind].Cf)

            bestKDr.append(MODEL_BANK[best_ind].Dr)
            bestKDf.append(MODEL_BANK[best_ind].Df)




        Brs_preds.append(np.mean(bestKBr))
        Bfs_preds.append(np.mean(bestKBf))

        Crs_preds.append(np.mean(bestKCr))
        Cfs_preds.append(np.mean(bestKCf))

        Drs_preds.append(np.mean(bestKDr))
        Dfs_preds.append(np.mean(bestKDf))

        smoothing_K = 10

        changing_run_model.Br = Brs_preds[-1]
        changing_run_model.Bf = Bfs_preds[-1]

        changing_run_model.Cr = Crs_preds[-1]
        changing_run_model.Cf = Cfs_preds[-1]

        changing_run_model.Dr = np.mean(np.array(Drs_preds)[-smoothing_K:])
        changing_run_model.Df = np.mean(np.array(Dfs_preds)[-smoothing_K:])

        changing_params['Bf'], changing_params['Br'] = changing_run_model.Bf, changing_run_model.Br
        changing_params['Cf'], changing_params['Cr'] = changing_run_model.Cf, changing_run_model.Cr
        changing_params['Df'], changing_params['Dr'] = changing_run_model.Df, changing_run_model.Dr


        MU_pred =(changing_run_model.Dr  + changing_run_model.Df ) / (9.81 * params['mass'])
        MU_preds.append(MU_pred)

    # if idt > 12:
    #     MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))
    #
    #     Nf = model.lf/(model.lf +  model.lr) * model.mass * 9.81
    #     Nr = model.lr/(model.lf +  model.lr) * model.mass * 9.81
    #     MU_pred = compute_mu_pred(Ffy_preds, Fry_preds, Nf, Nr, idt, window_size=50)
    #     MU_preds.append(MU_pred)

    # In your main loop:
    if idt > 0:
        eval_start = tm.time()
        predictions = evaluate_models_vectorized(MODEL_BANK, states[:, idt], inputs[:, idt], Ts, pool)
        eval_time = tm.time() - eval_start
        print(f"Model evaluation time: {eval_time:.3f}s")

        # Calculate errors for all models at once
        errors = np.mean((predictions[:, 0:4] - states[0:4, idt + 1]) ** 2, axis=1)

        # Update error windows using numpy operations
        error_windows = np.roll(error_windows, -1, axis=1)
        error_windows[:, -1] = errors
        window_count = min(window_count + 1, N_AC_STEPS)

        # Make decision if we have enough data points
        if window_count >= N_AC_STEPS:
            avg_errors = np.mean(error_windows, axis=1)
            new_model_idx = np.argmin(avg_errors)
            ind_best_KM = avg_errors.argsort()[:10]


    # forward sim to predict over the horizon
    hstates[:,0] = x0
    hstates2[:,0] = x0
    for idh in range(horizon):
        x_next, dxdt_next = model.sim_continuous(hstates[:,idh], umpc[:,idh].reshape(-1,1), [0, Ts])
        hstates[:,idh+1] = x_next[:,-1]
        hstates2[:,idh+1] = xmpc[:,idh+1]

    end = tm.time()
    print("iter: {}, cost: {:.5f}, time: {:.2f}".format(
        idt,
        fval,
        end - start    ))

    mean_cost = np.mean(fval_history)
    print(f"Mean cost at iter {idt}: {mean_cost:.3f}")

    if idt%500==0:
        # update plot
        LnS.set_xdata(states[0,:idt+1])
        LnS.set_ydata(states[1,:idt+1])
        #
        # LnR.set_xdata(xref[0,1:])
        # LnR.set_ydata(xref[1,1:])
        #
        # LnP.set_xdata(states[0,idt])
        # LnP.set_ydata(states[1,idt])
        #
        # LnH.set_xdata(hstates[0])
        # LnH.set_ydata(hstates[1])
        #
        # LnH2.set_xdata(hstates2[0])
        # LnH2.set_ydata(hstates2[1])

        LnFfy.set_xdata(time[:idt+1])
        LnFfy.set_ydata(Ffy[:idt+1])

        # LnFrx.set_xdata(time[:idt+1])
        # LnFrx.set_ydata(Frx[:idt+1])

        LnFry.set_xdata(time[:idt+1])
        LnFry.set_ydata(Fry[:idt+1])

        LnFfy_preds.set_xdata(time[:idt+1])
        LnFfy_preds.set_ydata(Ffy_preds[:idt+1])

        # LnFrx_preds.set_xdata(time[:idt+1])
        # LnFrx_preds.set_ydata(Frx_preds[:idt+1])

        LnFry_preds.set_xdata(time[:idt+1])
        LnFry_preds.set_ydata(Fry_preds[:idt+1])

        plt.pause(Ts/100)

plt.ioff()

#####################################################################
# save data

if SAVE_RESULTS:
    np.savez(
        '../data/DYN-NMPC-{}{}.npz'.format(SUFFIX, TRACK_NAME),
        time=time,
        states=states,
        dstates=dstates,
        inputs=inputs
        )

#####################################################################
# plots


# plot speed
# plt.figure()
# vel = np.sqrt(dstates[0,:]**2 + dstates[1,:]**2)
# plt.plot(time[:n_steps-horizon], vel[:n_steps-horizon], label='abs')
# plt.plot(time[:n_steps-horizon], states[3,:n_steps-horizon], label='vx')
# plt.plot(time[:n_steps-horizon], states[4,:n_steps-horizon], label='vy')
# plt.xlabel('time [s]')
# plt.ylabel('speed [m/s]')
# plt.grid(True)
# plt.legend()

# plot acceleration
# plt.figure()
# plt.plot(time[:n_steps-horizon], inputs[0,:n_steps-horizon])
# plt.xlabel('time [s]')
# plt.ylabel('PWM duty cycle [-]')
# plt.grid(True)

# plot mus
plt.figure()
plt.plot(MUs)
plt.plot(MU_preds)
plt.grid(True)
plt.ylabel('mu')

# plot steering angle
# plt.figure()
# plt.plot(time[:n_steps-horizon], inputs[1,:n_steps-horizon])
# plt.xlabel('time [s]')
# plt.ylabel('steering [rad]')
# plt.grid(True)

# # plot inertial heading
# plt.figure()
# plt.plot(time[:n_steps-horizon], states[2,:n_steps-horizon])
# plt.xlabel('time [s]')
# plt.ylabel('orientation [rad]')
# plt.grid(True)

# plt.figure()
# plt.plot(time[:n_steps-horizon], Dfs[:n_steps-horizon], label='Df')
# plt.plot(time[:n_steps-horizon], Drs[:n_steps-horizon], label='Dr')
# plt.plot(time[:n_steps-horizon], Dfs_preds[:n_steps-horizon], label='Df')
# plt.plot(time[:n_steps-horizon], Drs_preds[:n_steps-horizon], label='Dr')
# plt.xlabel('time [s]')
# plt.ylabel('mu*N [N]')
# plt.grid(True)
# plt.legend()

plt.show()