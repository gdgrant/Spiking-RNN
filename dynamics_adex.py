from imports import *
from parameters import par

def calculate_dynamics(prev_eps, st, input_data, spikes, psuedo_der, syn_x, syn_u, con_dict, eff_var, var_dict, t):
	""" Calculate the dynamics of the model
		prev_eps = the epsilons of the previous time step
		st       = dict holding all elements of model state
		x        = presynaptic from input
		z        = postsynaptic from recurrent
		z_prev   = presynaptic from recurrent
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

	# Make constant dictionary a shorter variable name for readability
	c = con_dict['adex']

	# Expand shapes to match [batch x presynaptic x postsynaptic]
	# in preparation for outer products and broadcasting
	z            = spikes[t, :, cp.newaxis, :]
	z_prev       = spikes[t-par['latency'], ...,cp.newaxis]
	z_prev_prev  = spikes[t-2*par['latency'], ..., cp.newaxis]
	x	         = input_data[t, ...,cp.newaxis]
	x_prev       = input_data[t-par['latency'], ...,cp.newaxis]
	h            = psuedo_der[t, :, cp.newaxis, :]
	h_prev       = psuedo_der[t-par['latency'], ..., cp.newaxis] if par['full_derivative'] else 0.

	# Cache common or unwieldy terms for readability and efficiency
	dt_over_C      = c['dt']/c['C']
	dt_g_over_C    = dt_over_C*c['g']
	dt_over_tau    = c['dt']/c['tau']
	dt_a_over_tau  = dt_over_tau*c['a']
	one_minus_beta = 1 - c['beta']

	d_eff_weights_raw_weights = ((var_dict['W_rnn'] >= 0) * con_dict['EI_vector'][:,cp.newaxis])[cp.newaxis,...]
	d_inp_weights_raw_weights = (var_dict['W_in'] >= 0)[cp.newaxis,:,:]
	
	one_minus_z           = 1. - z
	one_minus_z_dt_over_C = one_minus_z * dt_over_C
	one_minus_z_dt_mu_over_C = one_minus_z_dt_over_C * c['mu']

	# Use full derivative of voltage in gradient
	exp_v_minus_one = cp.exp((v-c['V_T'])/c['D'])-1
	#exp_v_minus_one = cp.minimum(1., exp_v_minus_one)
	exp_v_minus_one = cp.clip(exp_v_minus_one, -1., 1.)
	#exp_v_minus_one = cp.ones_like(exp_v_minus_one)
	eps_dyn_v = exp_v_minus_one

	adex_voltage_dvdv = one_minus_z*(1 + dt_g_over_C*eps_dyn_v)


	# Set up new epsilon recording
	eps = {}
	eps['inp'] = {'prev_v': prev_eps['inp']['prev_v']}
	eps['rec'] = {'prev_v': prev_eps['rec']['prev_v']}

	### Update input epsilons
	eps['inp']['v'] = \
		  prev_eps['inp']['v']  * adex_voltage_dvdv \
		- prev_eps['inp']['w']  * one_minus_z_dt_over_C \
		+ prev_eps['inp']['ia'] * one_minus_z_dt_mu_over_C

	eps['inp']['w'] = \
		  prev_eps['inp']['v']  * (dt_a_over_tau + c['b']*h) \
		+ prev_eps['inp']['w']  * (1 - dt_over_tau)

	eps['inp']['ia'] = \
		  prev_eps['inp']['ia'] * c['beta'] \
		+ one_minus_beta * x * d_inp_weights_raw_weights

	### Update recurrent epsilons
	eps['rec']['v'] = \
		  prev_eps['rec']['v'] * adex_voltage_dvdv \
		- prev_eps['rec']['w'] * one_minus_z_dt_over_C \
		+ prev_eps['rec']['ir'] * one_minus_z_dt_mu_over_C


	eps['rec']['w'] = \
		  prev_eps['rec']['v'] * (dt_a_over_tau + c['b']*h) \
		+ prev_eps['rec']['w'] * (1 - dt_over_tau)

	eps['rec']['ir'] = \
		  prev_eps['rec']['ir'] * c['beta'] \
		+ prev_eps['rec']['sx'] * one_minus_beta * eff_var['W_rnn'][cp.newaxis,:,:] * syn_u * z_prev \
		+ prev_eps['rec']['su'] * one_minus_beta * eff_var['W_rnn'][cp.newaxis,:,:] * syn_x * z_prev \
		+ one_minus_beta * syn_u * syn_x * z_prev * d_eff_weights_raw_weights

	eps['rec']['sx'] = \
		  prev_eps['rec']['sx'] * (1 - con_dict['alpha_std'] - syn_u*z_prev) \
		- prev_eps['rec']['su'] * syn_x * z_prev

	eps['rec']['su'] = \
		  prev_eps['rec']['su'] * (1 - con_dict['alpha_stf'] - con_dict['U']*z_prev)

	### Second-order terms
	"""
	# dI/dz * dZ/dV
	term_I = one_minus_beta * h_prev * syn_x * syn_u * eff_var['W_rnn'][cp.newaxis,:,:]
	# dSx/dz * dZ/dV
	term_Sx = -syn_x * syn_u * h_prev
	# dSu/dz * dZ/dV
	term_Su = con_dict['U'] * (1 - syn_u) * h_prev

	eps['rec']['v'] += cp.einsum('bij,bjk->bik', prev_eps['rec']['prev_v'][0], term_I + term_Sx + term_Su)
	eps['inp']['v'] += cp.einsum('bij,bjk->bik', prev_eps['inp']['prev_v'][0], term_I + term_Sx + term_Su)
	"""

	#eps['inp']['v'] += \
	#	prev_eps['inp']['prev_v'][0] * (term_I + term_Sx + term_Su)

	"""
	eps['rec']['v'] += \
		prev_eps['rec']['prev_v'][0] * h_prev \
		* ((-1 + one_minus_beta * eff_var['W_rnn'][cp.newaxis,:,:]) * syn_x * syn_u - con_dict['U'] * (1 - syn_u))
	"""

	### Second-order corrections to recurrent epsilons
	"""
	term1_z = one_minus_beta * h_prev * one_minus_z_dt_mu_over_C * syn_x_prev * syn_u_prev * z_prev_prev * d_eff_weights_raw_weights
	term1_x = one_minus_beta * h_prev * one_minus_z_dt_mu_over_C * x_prev
	term2 = one_minus_beta * syn_u  * syn_x * eff_var['W_rnn'][cp.newaxis,:,:]

	eps['rec']['ir'] += cp.einsum('bij,bjk->bik', term1_z, term2)
	eps['inp']['ia'] += cp.einsum('bij,bjk->bik', term1_x, term2)
	"""

	return eps
