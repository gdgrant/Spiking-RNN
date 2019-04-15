import numpy as np
from itertools import product
from math import trunc
print('\n--> Loading parameters...')

global par
par = {

	# Context
	'save_dir'              : './savedir/',
	'save_fn'               : 'adex_testing',

	# Training environment
	'batch_size'            : 256,
	'iterations'            : 1000,
	'cell_type'             : 'adex',   # 'lif', 'adex'

	# Network shape
	'num_motion_tuned'      : 24,
	'num_fix_tuned'         : 4,
	'num_rule_tuned'        : 0,
	'num_receptive_fields'  : 1,
	'num_motion_dirs'       : 8,
	'n_hidden'              : 100,
	'n_output'              : 3,

	# EI setup
	'EI_prop'               : 1.,
	'balance_EI'            : True,

	# AdEx parameters
	'exc_model'             : 'RS',
	'inh_model'             : 'cNA',
	'current_divider'       : 3e6,

	# Adam parameters
	'adam_beta1'            : 0.9,
	'adam_beta2'            : 0.999,
	'adam_epsilon'          : 1e-8,

	# Noise and weight scaling values
	'input_gamma'           : 0.08,
	'rnn_gamma'             : 0.04,
	'output_gamma'          : 0.08,
	'noise_rnn_sd'          : 0.05,
	'noise_in_sd'           : 0.2,

	# Timing constants
	'dt'                    : 1,
	'membrane_constant'     : 100,
	'output_constant'       : 20,

	# Task setup
	'task'                  : 'oic',
	'kappa'                 : 2.0,
	'tuning_height'         : 100.0,
	'response_multiplier'   : 1.,
	'num_rules'             : 1,
	'fixation_on'           : True,

	# Task timings
	'dead_time'             : 100,
	'fix_time'              : 200,
	'sample_time'           : 200,
	'delay_time'            : 300,
	'test_time'             : 200,
	'mask_time'             : 40,
}


def update_parameters(updates):
	for k in updates.keys():
		print(k.ljust(24), ': {}'.format(updates[k]))
		par[k] = updates[k]
	update_dependencies()


def update_dependencies():

	# Set up trial length and number of time steps
	par['trial_length'] = par['dead_time'] + par['fix_time'] \
		+ par['sample_time'] + par['delay_time'] + par['test_time']
	par['num_time_steps'] = par['trial_length'] // par['dt']

	# Network input and EI sizes
	par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] \
	+ par['num_fix_tuned'] + par['num_rule_tuned']
	par['n_EI'] = int(par['n_hidden']*par['EI_prop'])

	# Network initializations
	par['h_init_init']  = np.zeros([1,par['n_hidden']]).astype(np.float32)

	par['W_in_init']	= np.random.gamma(par['input_gamma'],  scale=1.0, size=[par['n_input'],  par['n_hidden']]).astype(np.float32)
	par['W_out_init']	= np.random.gamma(par['output_gamma'], scale=1.0, size=[par['n_hidden'], par['n_output']]).astype(np.float32)
	par['W_rnn_init']	= np.random.gamma(par['rnn_gamma'],    scale=1.0, size=[par['n_hidden'], par['n_hidden']]).astype(np.float32)

	if par['balance_EI']:
		par['W_rnn_init'][par['n_EI']:,:par['n_EI']] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=par['W_rnn_init'][par['n_EI']:,:par['n_EI']].shape).astype(np.float32)
		par['W_rnn_init'][:par['n_EI'],par['n_EI']:] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=par['W_rnn_init'][:par['n_EI'],par['n_EI']:].shape).astype(np.float32)

	par['b_rnn_init']   = np.zeros([1, par['n_hidden']]).astype(np.float32)
	par['b_out_init']   = np.zeros([1, par['n_output']]).astype(np.float32)

	par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])
	par['W_rnn_init']  *= par['W_rnn_mask']

	par['EI_vector']    = np.ones(par['n_hidden']).astype(np.float32)
	par['EI_vector'][par['n_EI']:] *= -1
	par['EI_mask']      = np.diag(par['EI_vector'])

	par['dt_sec']       = par['dt']/1000
	par['alpha_neuron'] = np.float32(par['dt']/par['membrane_constant'])
	par['beta_neuron']  = np.float32(par['dt']/par['output_constant'])
	par['noise_rnn']    = np.float32(np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd'])
	par['noise_in']     = np.float32(np.sqrt(2/par['alpha_neuron'])*par['noise_rnn_sd'])

	### Adaptive-Expoential spiking
	if par['cell_type'] == 'adex':

		# Note that voltages are in units of mV and currents
		# are in units of mA.  When pulling from a table based in volts/amps,
		# multiply E, V_T, D, b, V_r, and Vth by 1000
		par['cNA'] = {
			'C'   : 59e-12,     'g'   : 2.9e-9,     'E'   : -62,
			'V_T' : -42,        'D'   : 3,          'a'   : 1.8e-9,
			'tau' : 16e-3,      'b'   : 61e-9,      'V_r' : -54,
			'Vth' : 20,         'dt'  : par['dt']/1000 }
		par['RS']  = {
			'C'   : 104e-12,    'g'   : 4.3e-9,     'E'   : -65,
			'V_T' : -52,        'D'   : 0.8,        'a'   : -0.8e-9,
			'tau' : 88e-3,      'b'   : 65e-9,      'V_r' : -53,
			'Vth' : 20,         'dt'  : par['dt']/1000 }

		par['adex'] = {}
		for (k0, v_exc), (k1, v_inh) in zip(par[par['exc_model']].items(), par[par['inh_model']].items()):
			assert(k0 == k1)
			par_matrix = np.ones([1,par['n_hidden']], dtype=np.float32)
			par_matrix[:,:int(par['n_hidden']*par['EI_prop'])] *= v_exc
			par_matrix[:,int(par['n_hidden']*par['EI_prop']):] *= v_inh
			par['adex'][k0] = par_matrix

		par['w_init'] = par['adex']['b']
		par['adex']['current_divider'] = par['current_divider']


update_dependencies()
print('--> Parameters loaded.\n')
