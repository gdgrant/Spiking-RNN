from imports import *
from parameters import par

def calculate_dynamics(prev_eps, x, z, z_prev, st, h, h_prev, con_dict, eff_var):
	""" Calculate the dynamics of the model
		prev_eps = the epsilons of the previous time step
		x        = presynaptic from input
		z        = postsynaptic from recurrent
		z_prev   = presynaptic from recurrent
		st       = dict holding all elements of model state
		h        = pseudoderivative of z
		h_prev   = h from latency time ago
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
	# v = cp.minimum(-20e-3, v)

	# Make constant dictionary a shorter variable name for readability
	c = con_dict['adex']

	# Prepare addition of dimensions to constants
	s = np.s_[:,cp.newaxis,:]

	# Expand shapes to match [batch x presynaptic x postsynaptic]
	# in preparation for outer products and broadcasting
	x            = x[:,:,cp.newaxis]
	z            = z[:,cp.newaxis,:]
	z_prev       = z_prev[:,:,cp.newaxis]
	h            = h[:,cp.newaxis,:]
	h_prev       = h_prev[:,cp.newaxis,:] if par['full_derivative'] else 0
	v            = v[:,cp.newaxis,:]

	# Apply necessary variable rules
	z_prev_EI = z_prev * con_dict['EI_vector'][cp.newaxis,:,cp.newaxis]

	# Cache common or unwieldy terms for readability and efficiency
	dt_over_C      = c['dt']/c['C'][s]
	dt_g_over_C    = dt_over_C*c['g'][s]
	dt_over_tau    = c['dt']/c['tau'][s]
	dt_a_over_tau  = dt_over_tau*c['a'][s]
	one_minus_beta = 1 - c['beta']

	one_minus_z           = 1. - z
	one_minus_z_dt_over_C = one_minus_z * dt_over_C

	if par['dv_approx']:
		# Use approximation of exponential of voltage in gradient
		lin_v = (v - c['V_T'][s])/c['D'][s]
		eps_dyn_v = lin_v
	else:
		# Use full derivative of voltage in gradient
		exp_v_minus_one = cp.exp((v-c['V_T'][s])/c['D'][s])-1
		exp_v_minus_one = cp.minimum(1., exp_v_minus_one)
		eps_dyn_v = exp_v_minus_one

	# Set up epsilon recording
	eps = {}

	var_sets = ['inp', 'rec']
	zeds_pre = [x, z_prev]
	for v, z_i in zip(var_sets, zeds_pre):
		eps[v] = {}

		# Select context-dependent variables
		var_name  = 'W_in' if v=='inp' else 'W_rnn'
		stp_alpha = con_dict['alpha_std'] if v=='rec' else 0.
		stp_U     = con_dict['U'] if v=='rec' else 0.

		# Calculate eps_V
		eps[v]['v'] = \
			  prev_eps[v]['v'] * one_minus_z*(1+dt_g_over_C*eps_dyn_v) \
			- prev_eps[v]['w'] * one_minus_z_dt_over_C \
			+ prev_eps[v]['i'] * one_minus_z_dt_over_C * 1e-12

		# Calculate eps_w
		eps[v]['w'] = \
			  prev_eps[v]['v'] * (dt_a_over_tau + c['b'][s]*h) \
			+ prev_eps[v]['w'] * (1 - dt_over_tau)

		# Calculate eps_I
		eps[v]['i'] = \
			  prev_eps[v]['i']  * c['beta'] \
			+ prev_eps[v]['sx'] * one_minus_beta * syn_u[v] * eff_var[var_name][cp.newaxis,:,:] * z_i \
			+ prev_eps[v]['su'] * one_minus_beta * syn_x[v] * eff_var[var_name][cp.newaxis,:,:] * z_i \
			+ one_minus_beta * syn_x[v] * syn_u[v] * z_i \
			+ h_prev * one_minus_beta * syn_u[v] * syn_x[v] * eff_var[var_name][cp.newaxis,:,:]

		# Calculate eps_syn_x
		eps[v]['sx'] = \
			  prev_eps[v]['sx'] * (1 - stp_alpha - syn_u[v]*z_i) \
			- prev_eps[v]['su'] * syn_x[v] * z_i \
			+ h_prev * syn_u[v] * syn_x[v]

		# Calculate eps_syn_u
		eps[v]['su'] = \
			  prev_eps[v]['su'] * (1 - stp_alpha - stp_U*z_i) \
			  + h_prev * stp_U * (1 - syn_u[v])

	return eps
