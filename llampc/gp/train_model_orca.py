"""	Train a GP model for error discrepancy between kinematic and dynamic models.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import time
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from llampc.utils.plots import plot_true_predicted_variance

#####################################################################
# load data

SAVE_MODELS = True
SAVE_PARAMS = True

VARIDXs = [3,4,5]
state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']

def load_data(CTYPE, TRACK_NAME, VARIDX, xscaler=None, yscaler=None):
	data_dyn = np.load('../data/DYN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	data_kin = np.load('../data/KIN-{}-{}.npz'.format(CTYPE, TRACK_NAME))
	y_all = data_dyn['states'][:6,1:N_SAMPLES+1] - data_kin['states'][:6,1:N_SAMPLES+1]
	# y_all = data_dyn['states'][:6,1:N_SAMPLES+1] - data_dyn['states'][:6,1:N_SAMPLES+1]
	# print(N_SAMPLES)
	# print(data_kin['inputs'][:,:N_SAMPLES].T.shape)
	# print(data_kin['states'][6,:N_SAMPLES].reshape(1,-1).T.shape)
	# print(data_dyn['states'][3:6,:N_SAMPLES].T.shape)
	x = np.concatenate([
		data_kin['inputs'][:,:N_SAMPLES].T,
		data_kin['states'][6,:N_SAMPLES].reshape(1,-1).T,
		data_dyn['states'][3:6,:N_SAMPLES].T],
		axis=1)
	y = y_all[VARIDX].reshape(-1,1)

	if xscaler is None or yscaler is None:
		xscaler = StandardScaler()
		yscaler = StandardScaler()
		xscaler.fit(x)
		yscaler.fit(y)
		return xscaler.transform(x), yscaler.transform(y), xscaler, yscaler
	else:
		return xscaler.transform(x), yscaler.transform(y)

first = True
for VARIDX in VARIDXs :
	N_SAMPLES = 305
	filename = 'orca/{}gp.pickle'.format(state_names[VARIDX])
	filename_params = 'orca/{}gp_params.pickle'.format(state_names[VARIDX])

	x_train, y_train, xscaler, yscaler = load_data('PP', 'ETHZMobil', VARIDX)

	#####################################################################
	# train GP model

	k1 = 1.0*RBF(
		length_scale=np.ones(x_train.shape[1]),
		length_scale_bounds=(1e-5, 1e5),
		)
	k2 = ConstantKernel(0.1)
	kernel = k1 + k2
	# print(kernel)
	if first :
		model = GaussianProcessRegressor(
			alpha=1e-6, 
			kernel=kernel, 
			normalize_y=True,
			# optimizer=None,
			n_restarts_optimizer=1,
			)
		# first = False
	else :
		print("Tuned params : ", model.kernel_.get_params())
		kernel.set_params(**(model.kernel_.get_params()))
		model = GaussianProcessRegressor(
			alpha=1e-6, 
			kernel=kernel, 
			normalize_y=True,
			optimizer=None,
			# n_restarts_optimizer=0,
			)

	# model = SGDRegressor(
	# 	alpha=1e-6, 
	# 	kernel=kernel, 
	# 	normalize_y=True,
	# 	n_restarts_optimizer=10,
	# 	)
	start = time.time()
	model.fit(x_train, y_train)
	# print("Training done")
	end = time.time()
	print('training time: %ss' %(end - start))        
	print('final kernel: %s' %(model.kernel_))

	if SAVE_MODELS:
		with open(filename, 'wb') as f:
			pickle.dump((model, xscaler, yscaler), f)
	
	if SAVE_PARAMS :
		with open(filename_params, 'wb') as f:
			pickle.dump(model.kernel_.get_params(),f)
	#####################################################################
	# test GP model on training data

	y_train_mu, y_train_std = model.predict(x_train, return_std=True)
	y_train = yscaler.inverse_transform(y_train)
	y_train_mu = yscaler.inverse_transform(y_train_mu)
	y_train_std *= yscaler.scale_

	MSE = mean_squared_error(y_train, y_train_mu, multioutput='raw_values')
	R2Score = r2_score(y_train, y_train_mu, multioutput='raw_values')
	EV = explained_variance_score(y_train, y_train_mu, multioutput='raw_values')

	print('root mean square error: %s' %(np.sqrt(MSE)))
	print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_train.mean()))))
	print('R2 score: %s' %(R2Score))
	print('explained variance: %s' %(EV))

	#####################################################################
	# test GP model on validation data

	N_SAMPLES = 400
	x_test, y_test = load_data('NMPC', 'ETHZ', VARIDX, xscaler=xscaler, yscaler=yscaler)
	y_test_mu, y_test_std = model.predict(x_test, return_std=True)
	y_test = yscaler.inverse_transform(y_test)
	y_test_mu = yscaler.inverse_transform(y_test_mu)
	y_test_std *= yscaler.scale_

	MSE = mean_squared_error(y_test, y_test_mu, multioutput='raw_values')
	R2Score = r2_score(y_test, y_test_mu, multioutput='raw_values')
	EV = explained_variance_score(y_test, y_test_mu, multioutput='raw_values')

	print('root mean square error: %s' %(np.sqrt(MSE)))
	print('normalized mean square error: %s' %(np.sqrt(MSE)/np.array(np.abs(y_test.mean()))))
	print('R2 score: %s' %(R2Score))
	print('explained variance: %s' %(EV))

	#####################################################################
	# plot results

	plot_true_predicted_variance(
		y_train, y_train_mu, y_train_std, 
		ylabel='{} '.format(state_names[VARIDX]), xlabel='sample index'
		)

	plot_true_predicted_variance(
		y_test, y_test_mu, y_test_std, 
		ylabel='{} '.format(state_names[VARIDX]), xlabel='sample index'
		)

	plt.show()