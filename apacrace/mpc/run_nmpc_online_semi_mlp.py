"""	Nonlinear MPC using MLP for only learning tire forces
"""
fval_history = []

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

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'

import time as tm
import numpy as np
import casadi
# import _pickle as pickle
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from apacrace.params import ORCA
from apacrace.models import Dynamic, Kinematic6
from apacrace.gp.utils import loadGPModel, loadGPModelVars, loadMLPModel, loadTorchModel, loadTorchModelEq, loadTorchModelImplicit
from apacrace.tracks import ETHZ
from apacrace.mpc.planner import ConstantSpeed
from apacrace.mpc.gpmpc_torch import setupNLP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
import random
import os
import copy
import subprocess

import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
# matplotlib.use("pgf")  # Uses a LaTeX-compatible backend

import matplotlib.pylab as pylab



params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.rcParams['text.usetex'] = True


#####################################################################
# Tunable Params

GP_EPS_LEN = 410
mu_init = 1.
t_collect = 8.
LR = 0.002
BETA = 0.9

def dist(a,b) :
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# Dist covered, laps completed, Lap 0 time, Lap 1 time, Lap 2 time, Lap 3 time, Lap 4 time, Mean deviation, Track boundary violation time 
statistics = []

#####################################################################
# CHANGE THIS

ERROR_CORR = True
TRACK_CONS = False
RUN_NO = 'with_var_speeds' # From 'with_var_speeds', 'with_const_speeds' or 'without'
ITERS_EACH_STEP = 50
LOAD_MODEL = False
ACT_FN = 'relu'
SAFEGUARD = True
MAX_STEER = 0.34
VEHICLE_MODEL = 'Kinematic'
lambda_ = 10.
lambda_2 = 3.
ALPHA = 1.
torch.manual_seed(3)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

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

class DynamicModel(torch.nn.Module):
	def __init__(self, model, deltat = 0.01):
		"""
		In the constructor we instantiate four parameters and assign them as
		member parameters.
		"""
		super().__init__()
		if ACT_FN == 'relu' :
			self.act = torch.nn.ReLU()
		elif ACT_FN =='tanh' :
			self.act = torch.nn.Tanh()
		elif ACT_FN =='lrelu' :
			self.act = torch.nn.LeakyReLU()
		elif ACT_FN =='sigmoid' :
			self.act = torch.nn.Sigmoid()
		
		self.Rx = torch.nn.Sequential(torch.nn.Linear(1,1).to(torch.float64))
		
		self.Ry = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		self.Ry[0].weight.data.fill_(1.)
		# print(self.Ry[0].bias)
		self.Ry[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		self.Ry[2].weight.data.fill_(0.)
		self.Ry[2].bias.data.fill_(0.)
		# self.Ry[2].weight.data.fill_(1.)
		# print(self.Ry[0].bias)
		# print(self.Ry[0].weight)
		self.Fy = torch.nn.Sequential(torch.nn.Linear(1,6).to(torch.float64), \
					self.act, \
					torch.nn.Linear(6,1).to(torch.float64))
		
		self.Fy[0].weight.data.fill_(1.)
		self.Fy[2].weight.data.fill_(0.)
		self.Fy[2].bias.data.fill_(0.)

		# print(self.Ry[0].bias)
		self.Fy[0].bias.data = torch.arange(-.6,.6,(1.2)/6.).to(torch.float64)
		self.deltat = deltat
		self.model = model

	def forward(self, x, debug=False):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		# print(x.shape)
		# out = X
		deltatheta = x[:,1]
		theta = x[:,2]
		pwm = x[:,0]
		out = torch.zeros_like(x[:,3:6])
		# print(out)
		for i in range(2) :
			vx = (x[:,3] + out[:,0]).unsqueeze(1)
			vy = x[:,4] + out[:,1]
			w = x[:,5] + out[:,2]
			alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
			alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
			Ffy = self.Fy(alpha_f)[:,0]
			Fry = self.Ry(alpha_r)[:,0]
			Frx = self.Rx(vx**2)[:,0]
			Frx = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm + Frx
			
			if debug :
				print(Ffy,Fry,Frx)
			
			Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
			vx_dot = (Frx-Ffy*torch.sin(theta)+self.model.mass*vy*w)/self.model.mass
			vy_dot = (Fry+Ffy*torch.cos(theta)-self.model.mass*vx[:,0]*w)/self.model.mass
			w_dot = (Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/self.model.Iz
			out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat
		out2 = (out)
		return out2

#####################################################################
# default settings
def get_optimal_control_outer(pwm_ref,steer_ref,v,theta,x,curvature,model,nx=60,EPSILON=1e-2) :
	# theta = -theta
	# print(steer_ref,v,theta,x,curvature)
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], nx))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = x-EPSILON
		hd_left = v*math.sin(theta)
		hdd_left = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = np.linspace(model['min_pwm'], model['max_pwm'], nx)[ind[0]]
	min_steer = np.linspace(model['min_steer'], model['max_steer'], nx)[ind[1]]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control_inner(pwm_ref,steer_ref,v,theta,x,curvature,\
			      theta_,x_,curvature_,model,nx=60,ny=120,EPSILON=0.) :
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	print(curvature,curvature_)
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	if VEHICLE_MODEL=='Kinematic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	elif VEHICLE_MODEL=='Dynamic' :
		ax = ((model['Cm1']-model['Cm2']*v)*pwm - model['Cr0'] - model['Cr2']*v**2)/model['mass']
		h_left = -x-EPSILON
		hd_left = -v*math.sin(theta)
		hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
		
		h_right = x_-EPSILON
		hd_right = v*math.sin(theta_)
		hdd_right = ax*math.sin(theta_)+v**2*math.cos(theta_)*steer/L - v**2*curvature_
		
		cost = (steer-steer_ref)**2 + (pwm-pwm_ref)**2
		cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_**2*h_left)<0)*\
			(hdd_left+lambda_*hd_left+lambda_**2*h_left)**2
		cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_**2*h_right)<0)*\
			(hdd_right+lambda_*hd_right+lambda_**2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_**2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_**2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer

def get_optimal_control(pwm_ref,steer_ref,state_left,curvature_left,\
			      state_right,curvature_right,model,nx=60,ny=120,EPSILON=0.05) :
	print(state_right)
	if SAFEGUARD==False :
		return steer_ref
	min_steer = -MAX_STEER
	min_cost = 1000000
	
	# print(curvature,curvature_)
	L = model['lf'] + model['lr']
	
	pwm, steer = np.meshgrid(np.linspace(model['min_pwm'], model['max_pwm'], nx),\
				 np.linspace(model['min_steer'], model['max_steer'], ny))
	
	vx = state_right[2]
	vy = state_right[3]
	omega = state_right[4]
	
	alpha_f = steer - np.arctan2(omega*model['lf']+vy,vx)
	alpha_r = np.arctan2(omega*model['lr']-vy,vx)
	
	Ffy = model['Df']*np.sin(model['Cf']*np.arctan(model['Bf']*alpha_f))
	Fry = model['Dr']*np.sin(model['Cr']*np.arctan(model['Br']*alpha_r))
	v = np.sqrt(vx**2+vy**2)
	Frx = ((model['Cm1']-model['Cm2']*vx)*pwm)
	ax = (Frx-Ffy*np.sin(steer)+model['mass']*vy*omega)/model['mass']
	ay = (Fry+Ffy*np.cos(steer)-model['mass']*vx*omega)/model['mass']
	w_dot = (Ffy*model['lf']*np.cos(steer)-Fry*model['lr'])/model['Iz']
	
	# Ffy_ = model['Df']*np.sin(model['Cf']*np.arctan(model['Bf']*0.1))
	
	# ay_ = (Fry+Ffy_*np.cos(steer_)-model['mass']*vx*omega)/model['mass']
	
	# For right boundary
	d = state_right[0]
	theta = state_right[1]
	
	print("Curv : ", curvature_right)
	curvature_right = np.clip(curvature_right,-2,0.5)
	print("Theta : ", theta)
	h_right = d-EPSILON
	hd_right = vx*np.sin(theta) + vy*np.cos(theta)
	print("hd : ",hd_right+lambda_2*h_right)
	print((omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta)))
	hdd_right = (ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta))
	
	# For left boundary
	d = state_left[0]
	theta = state_left[1]
	# print("Curv : ", curvature_right)
	curvature_left = np.clip(curvature_left,-0.5,10)
	# print("Theta : ", theta)
	h_left = -d-EPSILON
	hd_left = -(vx*np.sin(theta) + vy*np.cos(theta))
	# print("hd : ",hd_right+lambda_2*h_right)
	# print((omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_right)*(vx*np.cos(theta)-vy*np.sin(theta)))
	hdd_left = -((ax*math.sin(theta)+ay*math.cos(theta)) + \
				(omega-(vx*np.cos(theta)-vy*np.sin(theta))*curvature_left)*(vx*np.cos(theta)-vy*np.sin(theta)))
	
	# h_left = -x-EPSILON
	# hd_left = -v*math.sin(theta)
	# hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
	
	cost = 100*(steer-steer_ref)**2 + (pwm-pwm_ref)**2
	cost += ALPHA*((hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)<0)*\
		(hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)**2
	cost += ALPHA*((hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)<0)*\
		(hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)**2
	
	ind = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
	min_pwm = pwm[ind]
	min_steer = steer[ind]
	feasibility_map = (hdd_left+lambda_*hd_left+lambda_*lambda_2*h_left)>=0.
	feasibility_map *= (hdd_right+lambda_*hd_right+lambda_*lambda_2*h_right)>=0.
	return min_steer, min_pwm, feasibility_map, pwm, steer


RUN_FOLDER = 'RUN_ONLINE_' + str(RUN_NO) + '_' + str(GP_EPS_LEN) + '/'

SIM_TIME = 10
SAMPLING_TIME = 0.02
HORIZON = 20
COST_Q = np.diag([1, 1])
COST_P = np.diag([0, 0])
COST_R = np.diag([5/1000, 1])

N_collect = int(t_collect/SAMPLING_TIME)
if not TRACK_CONS:
	SUFFIX = 'NOCONS-'+RUN_NO
else:
	SUFFIX = RUN_NO

#####################################################################
# load vehicle parameters

params = ORCA(control='pwm')
model = Dynamic(**params)
model_kin = Kinematic6(**params)

#####################################################################
# load track

TRACK_NAME = 'ETHZ'
track = ETHZ(reference='optimal', longer=True)
print(track.raceline[0][0])

#####################################################################
# load mlp models

MODEL_PATH = '../gp/orca/semi_mlp-v2.pickle'
model_ = DynamicModel(model)
if LOAD_MODEL :
	model_.load_state_dict(torch.load(MODEL_PATH))


model_Rx = loadTorchModelImplicit('Rx',model_.Rx)
model_Ry = loadTorchModelImplicit('Ry',model_.Ry)
model_Fy = loadTorchModelImplicit('Fy',model_.Fy)

models = {
	'Rx' : model_Rx,
	'Ry' : model_Ry,
	'Fy' : model_Fy,
	'act_fn' : ACT_FN
}
x_train = np.zeros((GP_EPS_LEN,2+3+1))

optimizer = torch.optim.SGD(model_.parameters(), lr=LR,momentum=BETA)
loss_fn = torch.nn.MSELoss()

#####################################################################
# extract data

Ts = SAMPLING_TIME
n_steps = int(SIM_TIME/Ts)
n_states = model.n_states
n_inputs = model.n_inputs
horizon = HORIZON

#####################################################################
# define controller

nlp = setupNLP(horizon, Ts, COST_Q, COST_P, COST_R, params, models, track, GP_EPS_LEN=GP_EPS_LEN, 
	track_cons=TRACK_CONS, error_correction=ERROR_CORR)

#####################################################################
# define load_data

def load_data(data_dyn, data_kin, VARIDX):
	y_all = (data_dyn['states'][:6,1:]-data_dyn['states'][:6,:-1]) #- data_kin['states'][:6,1:N_SAMPLES+1]
	x = np.concatenate([
		data_kin['inputs'][:,:-1].T,
		data_dyn['inputs'][1,:-1].reshape(1,-1).T,
		data_dyn['states'][3:6,:-1].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	return x, y

#####################################################################
# closed-loop simulation

# initialize
states = np.zeros([n_states+1, n_steps+1])
dstates = np.zeros([n_states, n_steps+1])
inputs = np.zeros([n_inputs, n_steps])
inputs_kin = np.zeros([n_inputs, n_steps])
time = np.linspace(0, n_steps, n_steps+1)*Ts
Ffy = np.zeros([n_steps+1])
Frx = np.zeros([n_steps+1])
Fry = np.zeros([n_steps+1])
hstates = np.zeros([n_states,horizon+1])
hstates2 = np.zeros([n_states,horizon+1])

Hs0 = []
Hs1 = []

Hs0_2 = []
Hs1_2 = []

data_dyn = {}
data_kin = {}
data_dyn['time'] = time
data_kin['time'] = time

projidx = 0
x_init = np.zeros(n_states)
x_init[0], x_init[1] = track.x_init, track.y_init
x_init[2] = track.psi_init
x_init[3] = track.vx_init
dstates[0,0] = x_init[3]
states[:n_states,0] = x_init
print('starting at ({:.1f},{:.1f})'.format(x_init[0], x_init[1]))
states_kin = np.zeros([7,n_steps+1])
states_kin[:,0] = states[:,0]


media_dir = "APACRace"
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

# main simulation loop
ref_speeds = []
Drs = []
Dfs = []
Drs_pred = []
Dfs_pred = []
Df_init = model.Df
Dr_init = model.Dr
h_outers = []
h_inners = []

MUs = []
MU_preds = []

dist_covered = 0
laps_completed = 0
lap_times = [0.,0.,0.,0.,0.]
cum_dists = [0.]

curr_cum_dist = 0.
for i in range(len(track.center_line[0])-1) :
	cum_dists.append(curr_cum_dist)
	curr_cum_dist += dist(track.center_line[:,i],track.center_line[:,i+1])

ref_errs = []
boundary_viol_time = 0.
v_factor = .9
alpha_f_max = 0.45
alpha_r_max = 0.45
for idt in range(n_steps-horizon):
	print("alpha maxes: ", alpha_f_max, alpha_r_max)
	start_g = tm.time()
	uprev = inputs[:,idt-1]
	x0 = states[:,idt]
	use_kinematic = True
	Drs.append(model.Dr)
	Dfs.append(model.Df)
	
	# load new experience into data_dyn and data_kin
	if idt > 0 : 
		start = tm.time()	
		min_ind = max(idt-GP_EPS_LEN-1,3)
		data_dyn['states'] = states[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['dstates'] = dstates[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_dyn['inputs'] = inputs[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		
		data_kin['states'] = states_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
		data_kin['inputs'] = inputs_kin[:,min_ind:min(idt-1,GP_EPS_LEN+1+min_ind)]
	
	
		end = tm.time()
		print("GP init time : ", end-start)
		
		y_trains = []
		for VARIDX in [3,4,5] :
			x_train, y_train = load_data(data_dyn,data_kin,VARIDX)
			y_trains.append(torch.tensor(y_train))
		y_train = torch.cat(y_trains,axis=1)
		x_train = torch.tensor(x_train)
	# Fine-tune the model
	if idt > 12 and 'without' not in RUN_NO:
		start = tm.time()	
		for i in range(ITERS_EACH_STEP) :
			# Zero your gradients for every batch!
			optimizer.zero_grad()
			outputs = model_(x_train[10:])
			loss = loss_fn(outputs, y_train[10:])
			loss.backward()
			# Adjust learning weights
			optimizer.step()
		end = tm.time()	
		print("Iter " + str(idt) + " loss : ", loss.item(), "time : ", end-start)

	model.Df, model.Dr = update_friction(model.Df, model.Dr, idt*Ts, "sudden")
	params['Df'], params['Dr'] = model.Df, model.Dr

	if idt > N_collect and 'without' not in RUN_NO:
		use_kinematic = False
		
	
	# planner based on BayesOpt
	prev_projidx = projidx
	if idt > N_collect and 'without' in RUN_NO:
		v_factor = 1.2
	if idt > N_collect and 'const_speeds' not in RUN_NO and 'without' not in RUN_NO:
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=(Dfs_pred[idt-1]+Drs_pred[idt-1])/(9.81*params['mass']),scale=v_factor)
	else :
		xref, projidx, v = ConstantSpeed(x0=x0[:2], v0=x0[3], track=track, N=horizon, Ts=Ts, projidx=projidx, curr_mu=mu_init,scale=v_factor)
	ref_speeds.append(v)
	fval_history.append(find_closest_point(x0[0],x0[1], track.raceline)[-1])

	if projidx > 656:
		if laps_completed > 0:
			lap_times[laps_completed] = idt * Ts  # - lap_times[laps_completed - 1]
			print(lap_times)
		else:
			lap_times[laps_completed] = idt * Ts
			print(lap_times)
		laps_completed += 1
		projidx = 0

	# print(projidx)
	# solve NLP
	start = tm.time()
	# projidx_inner, x_inner, theta_inner, curv_inner = GetCBFStateInner(x0=x0, track=track, projidx=projidx_inner)
	# projidx_outer, x_outer, theta_outer, curv_outer = GetCBFStateOuter(x0=x0, track=track, projidx=projidx_outer)
	# h_outers.append(x_outer)
	# h_inners.append(-x_inner)
	# if x_outer < 0 or x_inner>0 :
	# 	print("Boundary violated")
	# 	boundary_viol_time += Ts
	umpc, fval, xmpc = nlp.solve(x0=x0, xref=xref[:2,:], uprev=uprev, use_kinematic=use_kinematic,models=model_)

	end = tm.time()
	inputs[:,idt] = np.array([umpc[0,0], states[n_states,idt] + Ts*umpc[1,0]])
	print("iter: {}, cost: {:.5f}, time: {:.2f}".format(idt, fval, end-start))
	control_before = inputs[:,idt].copy()

	x_next, dxdt_next = model.sim_continuous(states[:n_states,idt], inputs[:,idt].reshape(-1,1), [0, Ts])
	inputs_kin[:,idt] = inputs[:,idt]

	if (idt!=0) :
		inputs_kin[1,idt] = (inputs[1,idt] - inputs[1,idt-1])/Ts
	else :
		inputs_kin[1,idt] = (inputs[1,idt] - 0.)/Ts

	x_next_kin, dxdt_next_kin = model_kin.sim_continuous(states[:,idt], inputs_kin[:,idt].reshape(-1,1), [0, Ts])

	states_kin[:,idt+1] = x_next_kin[:,-1]
	
	states[:n_states,idt+1] = x_next[:,-1]
	states[n_states,idt+1] = inputs[1,idt]

	dstates[:,idt+1] = dxdt_next[:,-1]
	Ffy[idt+1], Frx[idt+1], Fry[idt+1], alpha_f_curr, alpha_r_curr = model.calc_forces(states[:,idt], inputs[:,idt],return_slip=True)

	# forward sim to predict over the horizon
	steer = states[n_states,idt]
	hstates[:,0] = x0[:n_states]
	hstates2[:,0] = x0[:n_states]

	for idh in range(horizon):
		steer = steer + Ts*umpc[1,idh]
		hinput = np.array([umpc[0,idh], steer])
		x_next, dxdt_next = model.sim_continuous(hstates[:n_states,idh], hinput.reshape(-1,1), [0, Ts])
		hstates[:,idh+1] = x_next[:n_states,-1]
		hstates2[:,idh+1] = xmpc[:n_states,idh+1]

	Hs0.append(copy.deepcopy(hstates[0]))
	Hs1.append(copy.deepcopy(hstates[1]))

	Hs0_2.append(copy.deepcopy(hstates2[0]))
	Hs1_2.append(copy.deepcopy(hstates2[1]))

	mean_cost = np.mean(fval_history)
	print(f"Mean cost at iter {idt}: {mean_cost:.3f}")


	alpha_f_max = min(0.6,max(np.abs(alpha_f_curr),alpha_f_max))
	alpha_r_max = min(0.6,max(np.abs(alpha_r_curr),alpha_r_max))
	alpha_f = torch.tensor(np.arange(-alpha_f_max,alpha_f_max,0.01)).unsqueeze(1)
	Ffy_pred = model_.Fy(alpha_f)[:,0].detach().numpy()
	Ffy_true = params['Df']*torch.sin(params['Cf']*torch.atan(params['Bf']*alpha_f))

	if idt < N_collect :
		Dfs_pred.append(mu_init*params['mass']*9.8*params['lr']/(params['lf']+params['lr']))
	else:
		Dfs_pred.append(np.max(Ffy_pred))

	alpha_r = torch.tensor(np.arange(-alpha_r_max,alpha_r_max,0.01)).unsqueeze(1)
	Fry_pred = model_.Ry(alpha_r)[:,0].detach().numpy()
	Fry_true = params['Dr']*torch.sin(params['Cr']*torch.atan(params['Br']*alpha_r))

	if idt < N_collect :
		Drs_pred.append(mu_init*params['mass']*9.8*params['lf']/(params['lf']+params['lr']))
	else :
		Drs_pred.append(np.max(Fry_pred))

	if idt < N_collect :
		MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))
		MU_preds.append(mu_init)
	else:
		MUs.append((model.Df + model.Dr) / (9.81 * params['mass']))
		MU_preds.append((Dfs_pred[-1] + Drs_pred[-1]) / (9.81 * params['mass']))


def update(idt):

    ax.set_title(f"Frame {idt}")  # Optional: Add frame counter

    LnS.set_xdata(states[0, :idt + 1])
    LnS.set_ydata(states[1, :idt + 1])

    LnP.set_xdata(states[0, idt] + dims[:, 0] * np.cos(states[2, idt]) - dims[:, 1] * np.sin(states[2, idt]))
    LnP.set_ydata(states[1, idt] + dims[:, 0] * np.sin(states[2, idt]) + dims[:, 1] * np.cos(states[2, idt]))

    LnH.set_xdata(Hs0[idt])
    LnH.set_ydata(Hs1[idt])

    LnH2.set_xdata(Hs0_2[idt])
    LnH2.set_ydata(Hs1_2[idt])

    plt.tight_layout()


# plots

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
plt.plot(time[:n_steps-horizon], Dfs_pred[:n_steps-horizon],color="#0B67B2",linewidth=4,linestyle="-", label='Predicted Df')
plt.plot(time[:n_steps-horizon], Drs_pred[:n_steps-horizon],color="#D44A1C",linewidth=4,linestyle="-", label='Predicted Dr')
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

# Dist covered, laps completed, Lap 0 time, Lap 1 time, Lap 2 time, Lap 3 time, Lap 4 time, Mean deviation, Track boundary violation time 
for i in range(len(lap_times)-1,0,-1) :
	if lap_times[i] != 0. :
		lap_times[i] = lap_times[i] - lap_times[i-1]
print(lap_times)
