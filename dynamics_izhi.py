from imports import *
from gpu_utils import *
from parameters import par

def calculate_dynamics(prev_eps, st, input_data, spikes, pseudo_der, syn_x, syn_u, con_dict, eff_var, var_dict, t):

	### Unpack state dict
	# v     = membrane voltage
	# w     = adaptation current
	# I     = presynaptic current
	# syn_x = neurotransmitter level
	# syn_u = neurotransmitter utilization
	v, syn_x, syn_u = st['v'], st['sx'], st['su']

	# Make constant dictionary a shorter variable name for readability
	c = con_dict['izhi']

	# Expand shapes to match [batch x presynaptic x postsynaptic]
	# in preparation for outer products and broadcasting
	z            = spikes[t, :, cp.newaxis, :]
	z_prev       = spikes[t-par['latency'], ...,cp.newaxis]
	z_prev_prev  = spikes[t-2*par['latency'], ..., cp.newaxis]
	x	         = input_data[t, ...,cp.newaxis]
	x_prev       = input_data[t-par['latency'], ...,cp.newaxis]
	h            = pseudo_der[t, :, cp.newaxis, :]
	h_prev       = pseudo_der[t-par['latency'], ..., cp.newaxis] if par['full_derivative'] else 0.

	# Cache common or unwieldy terms
	ab             = c['a'] * c['b']
	one_minus_beta = 1 - c['beta']
	one_minus_z    = 1. - z

	one_minus_z_mu   = one_minus_z * c['mu']
	one_minus_z_conV = cp.minimum(one_minus_z * (0.08 * v + 6), 1.)
	ab_plus_dh       = ab + c['d'] * h

	d_eff_weights_raw_weights = ((var_dict['W_rnn'] >= 0) * con_dict['EI_vector'][:,cp.newaxis])[cp.newaxis,...]
	d_eff_weights_raw_weights = con_dict['EI_vector'][cp.newaxis,:,cp.newaxis]

	# Set up new epsilon recording
	eps = {}
	eps['inp'] = {'prev_v': prev_eps['inp']['prev_v']}
	eps['rec'] = {'prev_v': prev_eps['rec']['prev_v']}

	### Update input epsilons
	eps['inp']['v'] = \
		  prev_eps['inp']['v']  * one_minus_z_conV \
		- prev_eps['inp']['w']  * one_minus_z \
		+ prev_eps['inp']['ia'] * one_minus_z_mu

	eps['inp']['w'] = \
		  prev_eps['inp']['v']  * ab_plus_dh \
		+ prev_eps['inp']['w']  * (1 - c['a'])

	#print(cp.mean(eps['inp']['v']), cp.mean(eps['inp']['w']))

	eps['inp']['ia'] = \
		  prev_eps['inp']['ia'] * c['beta'] \
		+ one_minus_beta * x

	### Update recurrent epsilons
	eps['rec']['v'] = \
		  prev_eps['rec']['v'] * one_minus_z_conV \
		- prev_eps['rec']['w'] * one_minus_z \
		+ prev_eps['rec']['ir'] * one_minus_z_mu

	eps['rec']['w'] = \
		  prev_eps['rec']['v'] * ab_plus_dh \
		+ prev_eps['rec']['w'] * (1 - c['a'])

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


	return eps
