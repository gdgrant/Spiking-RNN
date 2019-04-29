import numpy as np
from itertools import product
from math import trunc
print('\n--> Loading parameters...')

global par
par = {

	# Context
	'save_dir'              : './savedir/',
	'save_fn'               : 'lif_testing',

	# Training environment
	'batch_size'            : 1024,
	'iterations'            : 1000,
	'cell_type'             : 'lif',   # 'lif', 'adex'
	'learning_rate'			: 0.001,

	# Network shape
	'num_motion_tuned'      : 24,
	'num_fix_tuned'         : 4,
	'num_rule_tuned'        : 0,
	'num_receptive_fields'  : 1,
	'num_motion_dirs'       : 8,
	'n_hidden'              : 100,
	'n_output'              : 3,

	# Pseudo derivative
	'eta'					: 1.0,

	# EI setup
	'EI_prop'               : 0.8,
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
	'task'                  : 'dms',
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
	par['h_init_init']  = np.zeros([1,par['n_hidden']])

	par['W_in_init']	= np.random.gamma(par['input_gamma'],  scale=1.0, size=[par['n_input'],  par['n_hidden']])
	par['W_out_init']	= np.random.gamma(par['output_gamma'], scale=1.0, size=[par['n_hidden'], par['n_output']])
	par['W_rnn_init']	= np.random.gamma(par['rnn_gamma'],    scale=1.0, size=[par['n_hidden'], par['n_hidden']])

	if par['balance_EI']:
		par['W_rnn_init'][par['n_EI']:,:par['n_EI']] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=par['W_rnn_init'][par['n_EI']:,:par['n_EI']].shape)
		par['W_rnn_init'][:par['n_EI'],par['n_EI']:] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=par['W_rnn_init'][:par['n_EI'],par['n_EI']:].shape)

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

	### LIF spiking (max 40-50 Hz; 10-20 Hz for preferred dir)

	par['W_in_const']   = np.zeros((par['n_input'], par['n_hidden']))
	U = np.arange(0, 360, 13) #73
	print(U)
	# beta = 0.04
	# kappa = 2
	# beta = 0.04
	# kappa = 0.75
	beta = 1.5
	kappa = 15
	z = beta/np.exp(kappa)
	for i in range(73):
		y = z * np.exp(kappa*np.cos(np.radians(U - i*5)))
		par['W_in_const'][:,i] = y

	# import matplotlib.pyplot as plt
	# fig, ax = plt.subplots(2,1)
	# ax[0].imshow(par['W_in_const'],aspect='auto')
	# ax[1].imshow(par['W_in_init'],aspect='auto')
	# plt.show()


	### Adaptive-Expoential spiking
	# Note that voltages are in units of mV and currents
	# are in units of mA.  When pulling from a table based in volts/amps,
	# multiply E, V_T, D, b, V_r, and Vth by 1000
	par['cNA'] = {
		'C'   : 59e-12,     'g'   : 2.9e-9,     'E'   : -62,
		'V_T' : -42,        'D'   : 3,          'a'   : 1.8e-9,
		'tau' : 16e-3,      'b'   : 61e-9,      'V_r' : -54,
		'Vth' : 20,         'dt'  : par['dt_sec'] }
	par['RS']  = {
		'C'   : 104e-12,    'g'   : 4.3e-9,     'E'   : -65,
		'V_T' : -52,        'D'   : 0.8,        'a'   : -0.8e-9,
		'tau' : 88e-3,      'b'   : 65e-9,      'V_r' : -53,
		'Vth' : 20,         'dt'  : par['dt_sec'] }

	par['adex'] = {}
	for (k0, v_exc), (k1, v_inh) in zip(par[par['exc_model']].items(), par[par['inh_model']].items()):
		assert(k0 == k1)
		par_matrix = np.ones([1,par['n_hidden']])
		par_matrix[:,:int(par['n_hidden']*par['EI_prop'])] *= v_exc
		par_matrix[:,int(par['n_hidden']*par['EI_prop']):] *= v_inh
		par['adex'][k0] = par_matrix

	par['w_init'] = par['adex']['b']
	par['adex']['current_divider'] = par['current_divider']

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
