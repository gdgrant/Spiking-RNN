from imports import *


def calculate_dynamics(prev_eps, x, z, z_prev, st, h, con_dict, eff_var):
	""" Calculate the dynamics of the model
		prev_eps = the epsilons of the previous time step
		x        = presynaptic from input
		z        = postsynaptic from recurrent
		z_prev   = presynaptic from recurrent
		st       = dict holding all elements of model state
		h        = pseudoderivative of z
		con_dict = constants
		var_dict = variables
	"""

	### Unpack state dict
	# v     = membrane voltage
	# w     = adaptation current
	# I     = presynaptic current
	# syn_x = neurotransmitter level
	# syn_u = neurotransmitter utilization
	v, syn_x, syn_u = st['v'], st['sx'], st['su']

	# Clip membrane voltage for optimization purposes
	v = cp.minimum(-20e-3, v)

	# Make constant dictionary a shorter variable name for readability
	c = con_dict['adex']

	# Prepare addition of dimensions to constants
	s = np.s_[:,:,cp.newaxis]

	# Calculate matmul operations before any other changes or operations
	presyn_x    = (x @ eff_var['W_in'])[s]
	presyn_z    = (z_prev @ eff_var['W_rnn'])[s]

	# Expand variable sizes, to shape 
	# [batch x postsynaptic neurons x presynaptic neurons],
	# in preparation for outer products / broadcasting
	x      = x[:,cp.newaxis,:]
	z      = z[:,:,cp.newaxis]
	z_prev = z_prev[:,cp.newaxis,:]
	h      = h[:,:,cp.newaxis]
	v      = v[:,:,cp.newaxis]
	syn_x  = syn_x[:,:,cp.newaxis]
	syn_u  = syn_u[:,:,cp.newaxis]

	# Apply necessary variable rules
	z_prev_EI = z_prev * con_dict['EI_vector'][cp.newaxis,cp.newaxis,:]

	# Cache common or unwieldy terms for readability and efficiency
	dt_over_C     = 1. #c['dt']/c['C'][s] # <-- Make this constant more appropriate
	dt_g_over_C   = dt_over_C*c['g'][s]
	dt_over_tau   = c['dt']/c['tau'][s]
	dt_a_over_tau = dt_over_tau*c['a'][s]

	one_minus_z           = 1. - z
	one_minus_z_dt_over_C = one_minus_z * dt_over_C
	syn_x_syn_u           = syn_x * syn_u

	exp_v_minus_one = cp.exp((v-c['V_T'][s])/c['D'][s])-1
	exp_v_minus_one = cp.clip(exp_v_minus_one, -1., 1.)

	# Set up epsilon recording
	eps = {}

	var_sets = ['inp', 'rec']
	presyns  = [presyn_x, presyn_z]
	zeds_pre = [x, z_prev]
	for v, ps, z_i in zip(var_sets, presyns, zeds_pre):
		eps[v] = {}

		# Calculate eps_V
		eps[v]['v'] = \
			  prev_eps[v]['v'] * one_minus_z*(1+dt_g_over_C*exp_v_minus_one) \
			- prev_eps[v]['w'] * one_minus_z_dt_over_C \
			+ prev_eps[v]['i'] * one_minus_z_dt_over_C

		# Calculate eps_w
		eps[v]['w'] = \
			  prev_eps[v]['v'] * (dt_a_over_tau + c['b'][s]*h) \
			+ prev_eps[v]['w'] * (1 - dt_over_tau)

		# Calculate eps_I
		eps[v]['i'] = \
			  prev_eps[v]['i']  * c['beta'] \
			+ prev_eps[v]['sx'] * syn_u * ps \
			+ prev_eps[v]['su'] * syn_x * ps \
			+ syn_x_syn_u * z_i

		# Calculate eps_syn_x
		eps[v]['sx'] = \
			  prev_eps[v]['sx'] * (1 - con_dict['alpha_std'][s] - c['dt']*syn_u*ps) \
			- prev_eps[v]['su'] * c['dt'] * syn_x * ps \
			- c['dt'] * syn_x_syn_u * z_i

		# Calculate eps_syn_u
		eps[v]['su'] = \
			  prev_eps[v]['su'] * (1 - con_dict['alpha_stf'][s] + c['dt']*con_dict['U'][s]*ps) \
			- c['dt'] * con_dict['U'][s] * (1-syn_u) * z_i

	# print()
	# for v in var_sets:
	# 	for e in ['v', 'w', 'i', 'sx', 'su']:
	# 		print(v, e.ljust(2), '|', eps[v][e].min(), eps[v][e].max())
			
	return eps
