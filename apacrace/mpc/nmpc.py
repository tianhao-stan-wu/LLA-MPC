"""	Setup NLP in CasADi.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np
import casadi as cs

from apacrace.mpc.constraints import Boundary


class setupNLP:
    #
    # def __init__(self, horizon, Ts, Q, P, R, params, model, track, track_cons=False):
    #     self.horizon = horizon
    #     self.params = params
    #     self.model = model
    #     self.track = track
    #     self.track_cons = track_cons
    #
    #     n_states = model.n_states
    #     n_inputs = model.n_inputs
    #     xref_size = 2
    #
    #     # Casadi vaiables
    #     x0 = cs.SX.sym('x0', n_states, 1)
    #     xref = cs.SX.sym('xref', xref_size, horizon + 1)
    #     uprev = cs.SX.sym('uprev', 2, 1)
    #     x = cs.SX.sym('x', n_states, horizon + 1)
    #     u = cs.SX.sym('u', n_inputs, horizon)
    #     dxdtc = cs.SX.sym('dxdt', n_states, 1)
    #     model_params = cs.SX.sym('model_params', 6, 1)  # Added for tire parameters
    #
    #     if track_cons:
    #         eps = cs.SX.sym('eps', 2, horizon)
    #         Aineq = cs.SX.sym('Aineq', 2 * horizon, 2)
    #         bineq = cs.SX.sym('bineq', 2 * horizon, 1)
    #
    #     # sum problem objectives and concatenate constraints
    #     cost_tracking = 0
    #     cost_actuation = 0
    #     cost_violation = 0
    #
    #     cost_tracking += (x[:xref_size, -1] - xref[:xref_size, -1]).T @ P @ (x[:xref_size, -1] - xref[:xref_size, -1])
    #     constraints = x[:, 0] - x0
    #
    #     for idh in range(horizon):
    #         dxdt = model.casadi_parametric(x[:, idh], u[:, idh], dxdtc, model_params)  # Changed to parametric
    #         constraints = cs.vertcat(constraints, x[:, idh + 1] - x[:, idh] - Ts * dxdt)
    #
    #     for idh in range(horizon):
    #         # delta between subsequent time steps
    #         if idh == 0:
    #             deltaU = u[:, idh] - uprev
    #         else:
    #             deltaU = u[:, idh] - u[:, idh - 1]
    #
    #         cost_tracking += (x[:xref_size, idh + 1] - xref[:xref_size, idh + 1]).T @ Q @ (
    #                     x[:xref_size, idh + 1] - xref[:xref_size, idh + 1])
    #         cost_actuation += deltaU.T @ R @ deltaU
    #
    #         if track_cons:
    #             cost_violation += 1e6 * (eps[:, idh].T @ eps[:, idh])
    #
    #         constraints = cs.vertcat(constraints, u[:, idh] - params['max_inputs'])
    #         constraints = cs.vertcat(constraints, -u[:, idh] + params['min_inputs'])
    #         constraints = cs.vertcat(constraints, deltaU[1] - params['max_rates'][1] * Ts)
    #         constraints = cs.vertcat(constraints, -deltaU[1] + params['min_rates'][1] * Ts)
    #
    #         # track constraints
    #         if track_cons:
    #             constraints = cs.vertcat(constraints,
    #                                      Aineq[2 * idh:2 * idh + 2, :] @ x[:2, idh + 1] - bineq[2 * idh:2 * idh + 2,
    #                                                                                       :] - eps[:, idh])
    #
    #     cost = cost_tracking + cost_actuation + cost_violation
    #
    #     xvars = cs.vertcat(
    #         cs.reshape(x, -1, 1),
    #         cs.reshape(u, -1, 1),
    #     )
    #     if track_cons:
    #         xvars = cs.vertcat(
    #             xvars,
    #             cs.reshape(eps, -1, 1),
    #         )
    #
    #     pvars = cs.vertcat(
    #         cs.reshape(x0, -1, 1),
    #         cs.reshape(xref, -1, 1),
    #         cs.reshape(uprev, -1, 1),
    #         model_params,  # Added model parameters
    #     )
    #     if track_cons:
    #         pvars = cs.vertcat(
    #             pvars,
    #             cs.reshape(Aineq, -1, 1),
    #             cs.reshape(bineq, -1, 1),
    #         )
    #
    #     nlp = {
    #         'x': xvars,
    #         'p': pvars,
    #         'f': cost,
    #         'g': constraints,
    #     }
    #     ipoptoptions = {
    #         'print_level': 0,
    #         'print_timing_statistics': 'no',
    #         'max_iter': 100,
    #     }
    #     self.options = {
    #         'expand': True,
    #         'print_time': False,
    #         'ipopt': ipoptoptions,
    #     }
    #     name = 'nmpc'
    #     self.problem = cs.nlpsol(name, 'ipopt', nlp, self.options)
    #
    # def solve(self, x0, xref, uprev, current_params=None):
    #     """
    #     Solve MPC problem
    #
    #     Args:
    #         x0: Initial state
    #         xref: Reference trajectory
    #         uprev: Previous control input
    #         current_params: Current tire parameters [Bf, Cf, Df, Br, Cr, Dr]
    #     """
    #     n_states = self.model.n_states
    #     n_inputs = self.model.n_inputs
    #     horizon = self.horizon
    #     track_cons = self.track_cons
    #
    #     # Update model parameters if provided
    #     if current_params is not None:
    #         self.model.Bf = current_params[0]
    #         self.model.Cf = current_params[1]
    #         self.model.Df = current_params[2]
    #         self.model.Br = current_params[3]
    #         self.model.Cr = current_params[4]
    #         self.model.Dr = current_params[5]
    #
    #     # Get current model parameters
    #     current_params = np.array([
    #         self.model.Bf,
    #         self.model.Cf,
    #         self.model.Df,
    #         self.model.Br,
    #         self.model.Cr,
    #         self.model.Dr
    #     ]).reshape(-1, 1)
    #
    #     # Print parameters being used
    #     print("\n=== Model Parameters Actually Used in Solver ===")
    #     print(f"Bf: {current_params[0][0]}")
    #     print(f"Cf: {current_params[1][0]}")
    #     print(f"Df: {current_params[2][0]}")
    #     print(f"Br: {current_params[3][0]}")
    #     print(f"Cr: {current_params[4][0]}")
    #     print(f"Dr: {current_params[5][0]}")
    #
    #     # track constraints
    #     if track_cons:
    #         Aineq = np.zeros([2 * horizon, 2])
    #         bineq = np.zeros([2 * horizon, 1])
    #         for idh in range(horizon):
    #             Ain, bin = Boundary(xref[:2, idh + 1], self.track)
    #             Aineq[2 * idh:2 * idh + 2, :] = Ain
    #             bineq[2 * idh:2 * idh + 2] = bin
    #     else:
    #         Aineq = np.zeros([0, 2])
    #         bineq = np.zeros([0, 1])
    #
    #     self.options['ipopt.warm_start_init_point'] = 'no'  # Force cold start
    #     self.options['ipopt.mu_init'] = 0.1  # Reset barrier parameter
    #
    #     arg = {}
    #     w_init = np.zeros(n_states * (horizon + 1) + n_inputs * horizon + 2 * horizon * track_cons)
    #     arg['x0'] = w_init
    #     arg['p'] = np.concatenate([
    #         x0.reshape(-1, 1),
    #         xref.T.reshape(-1, 1),
    #         uprev.reshape(-1, 1),
    #         current_params,  # Make sure this is included in the parameter vector
    #         Aineq.T.reshape(-1, 1),
    #         bineq.T.reshape(-1, 1),
    #     ])
    #     arg['lbx'] = -np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon + 2 * horizon * track_cons)
    #     arg['ubx'] = np.inf * np.ones(n_states * (horizon + 1) + n_inputs * horizon + 2 * horizon * track_cons)
    #     arg['lbg'] = np.concatenate(
    #         [np.zeros(n_states * (horizon + 1)), -np.inf * np.ones(horizon * (6 + 2 * track_cons))])
    #     arg['ubg'] = np.concatenate([np.zeros(n_states * (horizon + 1)), np.zeros(horizon * (6 + 2 * track_cons))])
    #
    #     res = self.problem(**arg)
    #     fval = res['f'].full()[0][0]
    #     xmpc = res['x'][:n_states * (horizon + 1)].full().reshape(horizon + 1, n_states).T
    #     umpc = res['x'][n_states * (horizon + 1):n_states * (horizon + 1) + n_inputs * horizon].full().reshape(horizon,
    #                                                                                                            n_inputs).T
    #     return umpc, fval, xmpc

######################################################## ORIGINAL ###############################################################
	def __init__(self, horizon, Ts, Q, P, R, params, model, track, track_cons=False):

		self.horizon = horizon
		self.params = params
		self.model = model
		self.track = track
		self.track_cons = track_cons

		n_states = model.n_states
		n_inputs = model.n_inputs
		xref_size = 2

		# Casadi vaiables
		x0 = cs.SX.sym('x0', n_states, 1)
		xref = cs.SX.sym('xref', xref_size, horizon+1)
		uprev = cs.SX.sym('uprev', 2, 1)
		x = cs.SX.sym('x', n_states, horizon+1)
		u = cs.SX.sym('u', n_inputs, horizon)
		dxdtc = cs.SX.sym('dxdt', n_states, 1)
		# pitch = cs.SX.sym('pitch', 1, 1)
		# roll = cs.SX.sym('roll', 1, 1)
		if track_cons:
			eps = cs.SX.sym('eps', 2, horizon)
			Aineq = cs.SX.sym('Aineq', 2*horizon, 2)
			bineq = cs.SX.sym('bineq', 2*horizon, 1)

		# sum problem objectives and concatenate constraints
		cost_tracking = 0
		cost_actuation = 0
		cost_violation = 0

		cost_tracking += (x[:xref_size,-1]-xref[:xref_size,-1]).T @ P @ (x[:xref_size,-1]-xref[:xref_size,-1])
		constraints = x[:,0] - x0

		track_cost_m = 0

		# Subsample centerline points (take every Nth point)
		subsample_factor = 8  # Adjust this number based on your needs
		center_x = cs.SX(track.center_line[0][::subsample_factor])
		center_y = cs.SX(track.center_line[1][::subsample_factor])

		for idh in range(horizon):
			dxdt = model.casadi(x[:,idh], u[:,idh], dxdtc)
			constraints = cs.vertcat( constraints, x[:,idh+1] - x[:,idh] - Ts*dxdt )

		for idh in range(horizon):

			# delta between subsequent time steps
			if idh==0:
				deltaU  = u[:,idh]-uprev
			else:
				deltaU = u[:,idh]-u[:,idh-1]

			cost_tracking += (x[:xref_size,idh+1]-xref[:xref_size,idh+1]).T @ Q @ (x[:xref_size,idh+1]-xref[:xref_size,idh+1])
			cost_actuation += deltaU.T @ R @ deltaU

			# print(dir(track))  # Shows available attributes/methods

			# distances = cs.sqrt((center_x - x[0, idh + 1]) ** 2 + (center_y - x[1, idh + 1]) ** 2)
			# distance_to_center = cs.mmin(distances)  # Get minimum distance
			# in_region1_1 = cs.logic_and(x[0, idh + 1] >= -0.74, x[0, idh + 1] <= -0.25)
			# in_region1_2 = cs.logic_and(x[1, idh + 1] >= -2, x[1, idh + 1] <= 0.5)
			#
			#
			# in_region2_1 = cs.logic_and(x[0, idh + 1] >= 0, x[0, idh + 1] <= 0.75)
			# in_region2_2 = cs.logic_and(x[1, idh + 1] >= -0.25, x[1, idh + 1] <= 1)
			#
			# in_region3_1 = cs.logic_and(x[0, idh + 1] >= 1, x[0, idh + 1] <= 1.5)
			# in_region3_2 = cs.logic_and(x[1, idh + 1] >= -1.5, x[1, idh + 1] <= -1)
			#
			# in_region1 = cs.logic_and(in_region1_1, in_region1_2)
			# in_region2 = cs.logic_and(in_region2_1, in_region2_2)
			# in_region3 = cs.logic_and(in_region3_1, in_region3_2)
			#
			# in_region = cs.logic_or(in_region1,in_region2)
			# in_region = cs.logic_or(in_region,in_region3)
			#
			#
			# track_cost_m += 1.5*((distance_to_center) ** 2) * in_region

			if track_cons:
				# cost_violation += 1e6 * (eps[:,idh].T @ eps[:,idh])
				cost_violation += 1e1 * (eps[:,idh].T @ eps[:,idh])


			constraints = cs.vertcat( constraints, u[:,idh] - params['max_inputs'] )
			constraints = cs.vertcat( constraints, -u[:,idh] + params['min_inputs'] )
			constraints = cs.vertcat( constraints, deltaU[1] - params['max_rates'][1]*Ts )
			constraints = cs.vertcat( constraints, -deltaU[1] + params['min_rates'][1]*Ts )

			# track constraints
			if track_cons:
				constraints = cs.vertcat( constraints, Aineq[2*idh:2*idh+2,:] @ x[:2,idh+1] - bineq[2*idh:2*idh+2,:] - eps[:,idh] )

		cost = cost_tracking + cost_actuation + cost_violation + track_cost_m

		xvars = cs.vertcat(
			cs.reshape(x,-1,1),
			cs.reshape(u,-1,1),
			)
		if track_cons:
			xvars = cs.vertcat(
				xvars,
				cs.reshape(eps,-1,1),
				)

		pvars = cs.vertcat(
			cs.reshape(x0,-1,1),
			cs.reshape(xref,-1,1),
			cs.reshape(uprev,-1,1),
			)
		if track_cons:
			pvars = cs.vertcat(
				pvars,
				cs.reshape(Aineq,-1,1),
				cs.reshape(bineq,-1,1),
				)
		# pvars = cs.vertcat(
		# 	pvars,
		# 	pitch,
		# 	roll,
		# )

		nlp = {
			'x': xvars,
			'p': pvars,
			'f': cost,
			'g': constraints,
			}
		ipoptoptions = {
			'print_level': 0,
			'print_timing_statistics': 'no',
			'max_iter': 100,
			}
		options = {
			'expand': True,
			'print_time': False,
			'ipopt': ipoptoptions,
		}
		name = 'nmpc'
		self.problem = cs.nlpsol(name, 'ipopt', nlp, options)
#
#
# ####################################################### ORIGINAL ###############################################################
	def solve(self, x0, xref, uprev):
		n_states = self.model.n_states
		n_inputs = self.model.n_inputs
		horizon = self.horizon
		track_cons = self.track_cons

		# track constraints
		if track_cons:
			Aineq = np.zeros([2*horizon,2])
			bineq = np.zeros([2*horizon,1])
			for idh in range(horizon):
				Ain, bin = Boundary(xref[:2,idh+1], self.track)
				Aineq[2*idh:2*idh+2,:] = Ain
				bineq[2*idh:2*idh+2] = bin
		else:
			Aineq = np.zeros([0,2])
			bineq = np.zeros([0,1])

		arg = {}
		arg['p'] = np.concatenate([
			x0.reshape(-1,1),
			xref.T.reshape(-1,1),
			uprev.reshape(-1,1),
			Aineq.T.reshape(-1,1),
			bineq.T.reshape(-1,1),
			])
		arg['lbx'] = -np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['ubx'] = np.inf*np.ones( n_states*(horizon+1) + n_inputs*horizon + 2*horizon*track_cons )
		arg['lbg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), -np.inf*np.ones(horizon*(6+2*track_cons))] )
		arg['ubg'] =  np.concatenate( [np.zeros(n_states*(horizon+1)), np.zeros(horizon*(6+2*track_cons))] )

		res = self.problem(**arg)
		fval = res['f'].full()[0][0]
		xmpc = res['x'][:n_states*(horizon+1)].full().reshape(horizon+1,n_states).T
		umpc = res['x'][n_states*(horizon+1):n_states*(horizon+1)+n_inputs*horizon].full().reshape(horizon,n_inputs).T
		return umpc, fval, xmpc

