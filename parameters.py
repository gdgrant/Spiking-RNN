from imports import *


### Model parameters

print('\n--> Loading parameters...')

global par
par = {

	# File context
	'save_dir'                : './savedir/',
	'savefn'                  : 'merge_testing',
	'save_data_files'         : False,
	'save_pdfs'               : False,

	# Training environment
	'batch_size'              : 64,
	'iterations'              : 2000,
	'learning_rate'           : 1e-12,
	'cell_type'               : 'adex',
	'optimizer'               : 'adam',

	# Optimization parameters
	'gamma_psd'               : 0.3,
	'L_spike_cost'            : 10.,
	'train_input_weights'     : True,
	'pseudo_th'               : 10e-3,
	'dv_approx'               : True,

	# Model architecture
	'use_stp'                 : True,
	'EI_prop'                 : 0.8,
	'balance_EI'              : True,
	'balance_EI_training'     : False,

	# Network shape
	'num_motion_tuned'        : 96,
	'num_fix_tuned'           : 8,
	'num_rule_tuned'          : 0,
	'num_receptive_fields'    : 1,
	'n_hidden'                : 500,
	'n_output'                : 3,

	# Timing constants (all in ms)
	'dt'                      : 1,
	'tau_hid'                 : 20,
	'tau_out'                 : 20,
	'latency'                 : [10,11],	# No latency = None

	# AdEx architecture
	'exc_model'               : 'RS',
	'inh_model'               : 'cNA',
	'conductance_mult'        : 50e-12,

	# Synaptic plasticity setup
	'tau_fast'                : 200,
	'tau_slow'                : 1500,
	'U_stf'                   : 0.15,
	'U_std'                   : 0.45,

	# Adam parameters
	'adam_beta1'              : 0.9,
	'adam_beta2'              : 0.999,
	'adam_epsilon'            : 1e-8,

	# Noise and weight scaling
	'input_gamma'             : 0.7,
	'rnn_gamma'               : 0.8,
	'output_gamma'            : 0.04,
	'rnn_cap'                 : 0.006,
	'noise_in_sd'             : 0.5,

	# Task setup
	'task'                    : 'dmc',
	'num_motion_dirs'         : 8,
	'kappa'                   : 2.,
	'tuning_height'           : 100.,
	'response_multiplier'     : 2.,
	'num_rules'               : 1,
	'fixation_on'             : True,

	# Task variable parameters
	'var_delay'               : True,
	'catch_prob'              : 0.1,

	# Task timings
	'dead_time'               : 20,
	'fix_time'                : 30,
	'sample_time'             : 150,
	'delay_time'              : 120,
	'test_time'               : 150,
	'mask_time'               : 20,

}


def make_weights_and_masks():

	# Make W_in and mask
	par['W_in_init'] = np.random.gamma(par['input_gamma'], \
		scale=1., size=[par['n_input'], par['n_hidden']])
	par['W_in_mask'] = np.zeros_like(par['W_in_init'])
	par['W_in_mask'][:,0:par['n_EI']:4] = 1.
	par['W_in_mask'][:,1:par['n_EI']:4] = 1.

	# Make W_out and mask
	par['W_out_init'] = np.random.uniform(-1., 1., size=[par['n_hidden'], par['n_output']])
	par['W_out_mask'] = np.ones_like(par['W_out_init'])

	# Make b_out and mask
	par['b_out_init'] = np.zeros([1, par['n_output']])
	par['b_out_mask'] = np.ones_like(par['b_out_init'])

	# Make W_rnn and mask
	if par['EI_prop'] == 1.:
		par['W_rnn_init'] = np.random.uniform(-par['rnn_gamma'], par['rnn_gamma'], size=[par['n_hidden'],par['n_hidden']])
	else:
		par['W_rnn_init'] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=[par['n_hidden'], par['n_hidden']])       
		if par['balance_EI']:
			par['W_rnn_init'][par['n_EI']:,:par['n_EI']] *= 2
			par['W_rnn_init'][:par['n_EI'],par['n_EI']:] *= 2

	par['W_rnn_mask']   = 1 - np.eye(par['n_hidden'])
	par['W_rnn_init']  *= par['W_rnn_mask']

	par['W_in_init'] *= par['conductance_mult']
	par['W_rnn_init'] *= par['conductance_mult']
	# par['W_out_init'] *= par['conductance_mult']

	# Remake W_in and mask if weight won't be trained
	if not par['train_input_weights']:

		par['W_in_const'] = np.zeros_like(par['W_in_init'])
		U = np.linspace(0, 360, par['n_input'])

		beta = 0.2
		kappa = 7.
		z = beta/np.exp(kappa)
		for i in range(0, par['n_hidden'], 4):
			if i < par['n_EI']:
				y = z * np.exp(kappa*np.cos(np.radians(U - i*(0.05*par['n_hidden']/par['n_input']))))
			else:
				y = z * np.exp(kappa*np.cos(np.radians(U - i*(0.17*par['n_hidden']/par['n_input']))))
			par['W_in_const'][:,i:i+2] = y[:,np.newaxis]
	
		par['W_in_init'] = 20*par['conductance_mult']*par['W_in_const']
		par['W_in_mask'] = np.ones_like(par['W_in_mask'])


def update_parameters(updates, verbose=True, update_deps=True):
	if verbose:
		print('\n--- Updates ----------------')
	for k in updates.keys():
		if verbose:
			print(k.ljust(24), ': {}'.format(updates[k]))
		par[k] = updates[k]
	if update_deps:
		update_dependencies()


def update_dependencies():

	# Var delay parameters
	par['var_delay_max'] = par['delay_time']
	par['var_delay_min'] = int(par['var_delay_max']/2)
	par['delay_times'] = np.array([par['delay_time']-20, par['delay_time'], par['delay_time']+20])

	# Set up trial length and number of time steps
	par['trial_length'] = par['dead_time'] + par['fix_time'] \
		+ par['sample_time'] + par['delay_times'][-1] + par['test_time']
	par['num_time_steps'] = par['trial_length'] // par['dt']

	# Network input and EI sizes
	par['n_input'] = par['num_motion_tuned']*par['num_receptive_fields'] \
	+ par['num_fix_tuned'] + par['num_rule_tuned']
	par['n_EI'] = int(par['n_hidden']*par['EI_prop'])

	# Generate EI vector and matrix
	par['EI_vector'] = np.ones(par['n_hidden'])
	par['EI_vector'][par['n_EI']:] *= -1
	par['EI_matrix'] = np.diag(par['EI_vector'])

	# Initialize weights and generate the associated masks
	make_weights_and_masks()

	# Generate latency indices
	if par['latency'] is None:
		par['latency_inds'] = 0
	else:
		latency_dt = [int(p//par['dt']) for p in par['latency']]
		if latency_dt[0] == latency_dt[1]:
			latency_dt[1] += 1

		par['latency_inds'] = np.random.randint(*latency_dt, size=par['n_hidden'])

	# Generate time constants and noise values
	par['dt_sec']       = par['dt']/1000
	par['alpha_neuron'] = par['dt']/par['tau_hid']
	par['noise_in']     = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']

	### STP
	if par['use_stp']:

		par['alpha_stf'] = np.ones([1,par['n_hidden'],1])
		par['alpha_std'] = np.ones([1,par['n_hidden'],1])
		par['U']         = np.ones([1,par['n_hidden'],1])

		par['syn_x_init'] = np.zeros([par['batch_size'],par['n_hidden'],1])
		par['syn_u_init'] = np.zeros([par['batch_size'],par['n_hidden'],1])

		for i in range(0,par['n_hidden'],2):
			par['alpha_stf'][:,i,:] = par['dt']/par['tau_slow']
			par['alpha_std'][:,i,:] = par['dt']/par['tau_fast']
			par['U'][:,i,:] = 0.15
			par['syn_x_init'][:,i,:] = 1
			par['syn_u_init'][:,i,:] = par['U'][:,i]

			par['alpha_stf'][:,i+1,:] = par['dt']/par['tau_fast']
			par['alpha_std'][:,i+1,:] = par['dt']/par['tau_slow']
			par['U'][:,i+1,:] = 0.45
			par['syn_x_init'][:,i+1,:] = 1
			par['syn_u_init'][:,i+1,:] = par['U'][:,i+1,:]

	# Spiking algorithms
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
		par['w_init']      = par['adex']['b']

		par['adex']['beta']  = np.exp(-par['dt']/par['tau_hid'])
		par['adex']['kappa'] = np.exp(-par['dt']/par['tau_out'])


	elif par['cell_type'] == 'lif':
		### LIF with Adaptive Threshold spiking
		par['lif'] = {
			'tau_m'     : 20e-3,
			'tau_a'     : 200e-3,
			'tau_o'     : 20e-3,
			'v_th'      : 0.61,
			'beta'      : 1.8
		}
		par['lif']['alpha'] = np.exp(-par['dt_sec']/par['lif']['tau_m'])
		par['lif']['rho']   = np.exp(-par['dt_sec']/par['lif']['tau_a'])
		par['lif']['kappa'] = np.exp(-par['dt_sec']/par['lif']['tau_o'])


update_dependencies()
print('--> Parameters loaded.\n')