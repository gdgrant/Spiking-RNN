import numpy as np
from itertools import product
from math import trunc
print('\n--> Loading parameters...')

global par
par = {

	# Context
	'save_dir'              : './savedir/',
	'save_fn'               : 'lif_testing',
	'optimizer'				: 'adam',

	# Training environment
	'batch_size'            : 256,
	'iterations'            : 100000,
	'cell_type'             : 'adex',   # 'lif', 'adex'
	'learning_rate'			: 2e-4,

	# Network shape
	'num_motion_tuned'      : 96,
	'num_fix_tuned'         : 16,
	'num_rule_tuned'        : 0,
	'num_receptive_fields'  : 1,
	'num_motion_dirs'       : 8,
	'n_hidden'              : 500,
	'n_output'              : 3,

	# Optimization parameters
	'gamma'					: 0.3,
	'L_spike_cost'			: 0.01,
	'train_input_weights'	: False,

	# EI setup
	'EI_prop'               : 0.8,
	'balance_EI'            : True,

	# AdEx parameters
	'exc_model'             : 'RS',
	'inh_model'             : 'cNA',
	'current_divider'       : 5e8,

	# Adam parameters
	'adam_beta1'            : 0.9,
	'adam_beta2'            : 0.999,
	'adam_epsilon'          : 1e-8,

	# Noise and weight scaling values
	'input_gamma'           : 0.002,
	'rnn_gamma'             : 0.008,
	'output_gamma'          : 0.05,
	'rnn_cap'				: 0.004,
	'noise_rnn_sd'          : 0.5,
	'noise_in_sd'           : 0.2,

	# Timing constants
	'dt'                    : 1,
	'membrane_constant'     : 100,
	'output_constant'       : 20,
	'latency'				: [5,15],	# No latency = None

	# Task setup
	'task'                  : 'dmc',
	'kappa'                 : 2.0,
	'tuning_height'         : 100.0,
	'response_multiplier'   : 1.,
	'num_rules'             : 1,
	'fixation_on'           : True,

	# Task timings
	'dead_time'             : 20,
	'fix_time'              : 50,
	'sample_time'           : 120,
	'delay_time'            : 60,
	'test_time'             : 120,
	'mask_time'             : 30,
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
	par['h_init_init']  = np.zeros([1,par['n_hidden']])

	par['W_in_init']	= np.random.gamma(par['input_gamma'],  scale=1.0, size=[par['n_input'],  par['n_hidden']])
	par['W_in_mask']	= np.ones([par['n_input'],par['n_hidden']])
	par['W_in_mask'][:,::2] = 0.

	# par['W_out_init']	= np.random.gamma(par['output_gamma'], scale=1.0, size=[par['n_hidden'], par['n_output']])
	par['W_out_init']	= np.random.uniform(-par['output_gamma'], par['output_gamma'], size=[par['n_hidden'], par['n_output']])

	if par['EI_prop'] == 1.:
		par['W_rnn_init'] = np.random.uniform(-par['rnn_gamma'], par['rnn_gamma'], size=[par['n_hidden'],par['n_hidden']])
	else:
		par['W_rnn_init'] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=[par['n_hidden'], par['n_hidden']])
		# par['W_rnn_init'] = np.random.uniform(0, par['rnn_gamma'], size=[par['n_hidden'],par['n_hidden']])
		if par['balance_EI']:
			par['W_rnn_init'][par['n_EI']:,:par['n_EI']] *= 2
			par['W_rnn_init'][:par['n_EI'],par['n_EI']:] *= 2
		# par['W_rnn_init'] = np.random.uniform(0, par['rnn_gamma'], size=[par['n_hidden'],par['n_hidden']])

	par['b_rnn_init']   = np.zeros([1, par['n_hidden']])
	par['b_out_init']   = np.zeros([1, par['n_output']])

	par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])
	par['W_rnn_init']  *= par['W_rnn_mask']

	par['EI_vector']    = np.ones(par['n_hidden'])
	par['EI_vector'][par['n_EI']:] *= -1
	par['EI_mask']      = np.diag(par['EI_vector'])

	par['dt_sec']       = par['dt']/1000
	par['alpha_neuron'] = par['dt']/par['membrane_constant']
	par['beta_neuron']  = par['dt']/par['output_constant']
	par['noise_rnn']    = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
	par['noise_in']     = np.sqrt(2/par['alpha_neuron'])*par['noise_rnn_sd']

	if par['latency'] is None:
		par['latency_inds'] = 0
	else:
		par['latency_inds'] = np.random.randint(*par['latency'], size=par['n_hidden'])

	### LIF spiking (max 40-50 Hz; 10-20 Hz for preferred dir)

	if not par['train_input_weights']:
		# import matplotlib.pyplot as plt
		# fig, ax = plt.subplots(2,1)
		# ax[0].imshow(par['W_in_init'],aspect='auto')

		par['W_in_const'] = np.zeros((par['n_input'], par['n_hidden']))
		U = np.linspace(0, 360, par['n_input'])

		beta = 0.2
		kappa = 7.
		z = beta/np.exp(kappa)
		for i in range(0, par['n_hidden'], 2):
			if i < par['n_EI']:
				y = z * np.exp(kappa*np.cos(np.radians(U - i*(0.2*par['n_hidden']/par['n_input']))))
			else:
				y = z * np.exp(kappa*np.cos(np.radians(U - i*(0.8*par['n_hidden']/par['n_input']))))
			par['W_in_const'][:,i] = y
		par['W_in_init'] = 0.1*par['W_in_const']

		# ax[1].imshow(par['W_in_const'],aspect='auto')
		# plt.show()


	par['adex'] = {}
	par['lif'] = {}

	if par['cell_type'] == 'adex':
		### Adaptive-Expoential spiking
		# Note that voltages are in units of V, A, and secs
		par['cNA'] = {
			'C'   : 59e-12,     'g'   : 2.9e-9,     'E'   : -62e-3,
			'V_T' : -42e-3,     'D'   : 3e-3,       'a'   : 1.8e-9,
			'tau' : 16e-3,      'b'   : 61e-12,     'V_r' : -54e-3,
			'Vth' : 0e-3,       'dt'  : par['dt']/1000 }
		par['RS']  = {
			'C'   : 104e-12,    'g'   : 4.3e-9,     'E'   : -65e-3,
			'V_T' : -52e-3,     'D'   : 0.8e-3,     'a'   : -0.8e-9,
			'tau' : 88e-3,      'b'   : 65e-12,     'V_r' : -53e-3,
			'Vth' : 0e-3,       'dt'  : par['dt']/1000 }

		for (k0, v_exc), (k1, v_inh) in zip(par[par['exc_model']].items(), par[par['inh_model']].items()):
			assert(k0 == k1)
			par_matrix = np.ones([1,par['n_hidden']])
			par_matrix[:,:par['n_EI']] *= v_exc
			par_matrix[:,par['n_EI']:] *= v_inh
			par['adex'][k0] = par_matrix

		# par['adex'] = par['RS']
		par['adex']['Vth'] = 0.
		par['adex']['dt']  = par['dt']/1000
		par['w_init'] = par['adex']['b']
		par['adex']['current_divider'] = par['current_divider']

		par['tau_i'] = 10e-3
		par['adex']['beta']	= np.exp(-par['dt_sec']/par['tau_i'])

		par['tau_o'] = 20e-3
		par['adex']['kappa'] = np.exp(-par['dt_sec']/par['tau_o'])


	elif par['cell_type'] == 'lif':
		### LIF with Adaptive Threshold spiking
		par['lif'] = {
			'tau_m'		: 20e-3,
			'tau_a'		: 200e-3,
			'tau_o'		: 20e-3,
			'v_th'		: 0.61,
			'beta'		: 1.8
		}
		par['lif']['alpha'] = np.exp(-par['dt_sec']/par['lif']['tau_m'])
		par['lif']['rho']   = np.exp(-par['dt_sec']/par['lif']['tau_a'])
		par['lif']['kappa']	= np.exp(-par['dt_sec']/par['lif']['tau_o'])

update_dependencies()
print('--> Parameters loaded.\n')
