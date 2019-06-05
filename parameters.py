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
	'plot_EI_testing'         : False,
	'loadfn'                  : './savedir/_testing_data_iter003500.pkl',
	'load_weights'            : False,

	# Training environment
	'batch_size'              : 64,
	'iterations'              : 20000,
	'learning_rate'           : 1e-3,
	'spike_model'             : 'izhi',
	'optimizer'               : 'adam',

	# Optimization parameters
	'gamma_psd'               : 0.3,
	'L_spike_cost'            : 10.,
	'train_input_weights'     : True,
	'pseudo_th'               : 10e-3,
	'dv_approx'               : True,

	# Model architecture
	'use_stp'                 : True,
	'full_derivative'         : True,
	'EI_prop'                 : 0.8,
	'balance_EI'              : True,
	'balance_EI_training'     : True,

	# Network shape
	'num_motion_tuned'        : 100,
	'num_fix_tuned'           : 5,
	'num_rule_tuned'          : 20,
	'num_receptive_fields'    : 1,
	'n_hidden'                : 500,
	'n_output'                : 3,

	# Timing constants (all in ms)
	'dt'                      : 1,
	'tau_hid'                 : 5,
	'tau_out'                 : 20,
	'latency'                 : 10,

	# AdEx architecture
	'exc_model'               : 'RS',
	'inh_model'               : 'cNA',
	'current_multiplier'      : 1e-9,

	# Synaptic plasticity setup
	'tau_fast'                : 200,
	'tau_slow'                : 1500,
	'U_stf'                   : 0.15,
	'U_std'                   : 0.45,

	# Adam parameters
	'adam_beta1'              : 0.9,
	'adam_beta2'              : 0.999,
	'adam_epsilon'            : 1e-8,
	'betagrad'				  : 0,

	# Noise and weight scaling
	'input_gamma'             : 0.3,
	'rnn_gamma'               : 0.15,
	'output_gamma'            : 0.04,
	'rnn_cap'                 : 0.006,
	'noise_in_sd'             : 8.,

	# Task setup
	'task'                    : 'dmc',
	'num_motion_dirs'         : 4,
	'kappa'                   : 2.,
	'tuning_height'           : 40.,# was 20
	'response_multiplier'     : 2.,
	'num_rules'               : 2,
	'fixation_on'             : True,

	# Task variable parameters
	'var_delay'               : False,
	'catch_prob'              : 0.1,

	# Task timings
	'dead_time'               : 50,
	'fix_time'                : 100,
	'sample_time'             : 150,
	'delay_time'              : 120,
	'test_time'               : 100,
	'mask_time'               : 80,

	'local_rate'			  : 5000.,
	'weight_decay'			  : 4e-8,

}

def load_custom_weights():

	if par['load_weights']:
		data = pickle.load(open(par['loadfn'], 'rb'))
		var_dict = data['weights']

		print('\nLoading custom variable initializations for {}.'.format(var_dict.keys()))
		for name, val in var_dict.items():
			par[name+'_init'] = val
		print('Custom variables loaded.\n')


def make_weights_and_masks():

	# Make W_in and mask
	par['W_in_init'] = np.random.gamma(par['input_gamma'], \
		scale=1., size=[par['n_input'], par['n_hidden']]).astype(np.float64)
	par['W_in_mask'] = np.ones_like(par['W_in_init'])
	#par['W_in_mask'][:,0:par['n_hidden']:4] = 1.
	#par['W_in_init'] *= par['W_in_mask']

	# Make W_out and mask
	par['W_out_init'] = np.random.uniform(-1., 1., size=[par['n_hidden'], par['n_output']]).astype(np.float64)
	par['W_out_mask'] = np.ones_like(par['W_out_init'])

	# Make b_out and mask
	par['b_out_init'] = np.zeros([1, par['n_output']], dtype=np.float64)
	par['b_out_mask'] = np.ones_like(par['b_out_init'])

	# Make W_rnn and mask
	if par['EI_prop'] == 1.:
		par['W_rnn_init'] = np.random.uniform(-par['rnn_gamma'], par['rnn_gamma'], size=[par['n_hidden'],par['n_hidden']]).astype(np.float64)
	else:
		par['W_rnn_init'] = np.random.gamma(par['rnn_gamma'], scale=1.0, size=[par['n_hidden'], par['n_hidden']]).astype(np.float64)
		if par['balance_EI']:
			par['W_rnn_init'][par['n_exc']:,:] *= 1.8
			par['W_rnn_init'][:,par['n_exc']:] *= 1.8

	par['W_rnn_mask']   = 1 - np.eye(par['n_hidden']).astype(np.float64)
	par['W_rnn_init']  *= par['W_rnn_mask']

	par['W_rnn_init'] = np.minimum(4., par['W_rnn_init']).astype(np.float64)


	# Remake W_in and mask if weight won't be trained
	if not par['train_input_weights']:

		par['W_in_const'] = np.zeros_like(par['W_in_init'], dtype=np.float64)
		U = np.linspace(0, 360, par['n_input']).astype(np.float64)
		beta = 4.
		kappa = 7.
		z = beta/np.exp(kappa)
		for i in range(0, par['n_hidden'], 4):
			if i < par['n_exc']:
				y = z * np.exp(kappa*np.cos(2*np.pi*(i/par['n_exc'] + U/360)))
			else:
				y = z * np.exp(kappa*np.cos(2*np.pi*(i/(par['n_hidden']-par['n_exc']) + U/360)))
			par['W_in_const'][:,i:i+1] = y[:,np.newaxis]

		par['W_in_init'] = par['W_in_const']
		par['W_in_mask'] = np.ones_like(par['W_in_mask'])

	#plt.imshow(par['W_in_init'],aspect='auto')
	#plt.colorbar()
	#plt.show()


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
	par['n_exc'] = int(par['n_hidden']*par['EI_prop'])

	# Generate EI vector and matrix
	par['EI_vector'] = np.ones(par['n_hidden'], dtype=np.float64)
	par['EI_vector'][par['n_exc']:] *= -1
	par['EI_matrix'] = np.diag(par['EI_vector']).astype(np.float64)

	par['exh_vector'] = np.ones(par['n_hidden'], dtype=np.float64)
	par['exh_vector'][par['n_exc']:] *= 0
	par['EI_mask_exh'] = np.diag(par['exh_vector']).astype(np.float64)

	par['inh_vector'] = np.ones(par['n_hidden'], dtype=np.float64)
	par['inh_vector'][:par['n_exc']] *= 0
	par['EI_mask_inh'] = np.diag(par['inh_vector']).astype(np.float64)

	# Initialize weights and generate the associated masks
	make_weights_and_masks()

	# Generate time constants and noise values
	par['dt_sec']       = par['dt']/1000
	par['alpha_neuron'] = par['dt']/par['tau_hid']
	#par['noise_in']     = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd']
	par['noise_in']     = par['noise_in_sd']

	### STP
	if par['use_stp']:

		par['alpha_stf'] = np.ones([1,par['n_hidden'],1], dtype=np.float64)
		par['alpha_std'] = np.ones([1,par['n_hidden'],1], dtype=np.float64)
		par['U']         = np.ones([1,par['n_hidden'],1], dtype=np.float64)

		par['syn_x_init'] = np.zeros([par['batch_size'],par['n_hidden'],1], dtype=np.float64)
		par['syn_u_init'] = np.zeros([par['batch_size'],par['n_hidden'],1], dtype=np.float64)

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
	par['lif']  = {}
	par['izhi'] = {}

	if par['spike_model'] == 'adex':
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
			par_matrix = np.ones([1,1,par['n_hidden']], dtype=np.float64)
			par_matrix[:,:,:par['n_exc']] *= v_exc
			par_matrix[:,:,par['n_exc']:] *= v_inh
			par['adex'][k0] = par_matrix

		# par['adex'] = par['RS']
		par['adex']['Vth'] = 0.
		par['adex']['dt']  = par['dt']/1000
		par['w_init']      = par['adex']['b']
		par['v_init']      = par['adex']['V_r']

		par['adex']['beta']  = np.exp(-par['dt']/par['tau_hid']).astype(np.float64)
		par['adex']['kappa'] = np.exp(-par['dt']/par['tau_out']).astype(np.float64)
		par['adex']['mu']    = par['current_multiplier']

	elif par['spike_model'] == 'izhi':
		### Izhikevich spiking
		# Note that voltages are in units of V, A, and secs
		par['RS'] = {
			'a'	: 0.02,
			'b'	: 0.2,
			'c'	: -65.,
			'd'	: 8. }

		par['FS']  = {
			'a'	: 0.1,
			'b'	: 0.2,
			'c'	: -65.,
			'd'	: 2. }

		for (k0, v_exc), (k1, v_inh) in zip(par['RS'].items(), par['FS'].items()):
			assert(k0 == k1)
			par_matrix = np.ones([1,1,par['n_hidden']], dtype=np.float64)
			par_matrix[:,:,:par['n_exc']] *= v_exc
			par_matrix[:,:,par['n_exc']:] *= v_inh
			par['izhi'][k0] = par_matrix

		# par['adex'] = par['RS']
		par['izhi']['Vth'] = 30.
		par['izhi']['dt']  = par['dt']/1000
		par['izhi']['V_r'] = par['izhi']['c']
		par['w_init']      = par['izhi']['b'] * par['izhi']['c']
		par['v_init']      = par['izhi']['c']

		par['izhi']['beta']  = np.exp(-par['dt']/par['tau_hid']).astype(np.float64)
		par['izhi']['kappa'] = np.exp(-par['dt']/par['tau_out']).astype(np.float64)
		par['izhi']['mu']    = 5.

	elif par['spike_model'] == 'lif':
		### LIF with Adaptive Threshold spiking
		par['lif'] = {
			'tau_m'     : 20e-3,
			'tau_a'     : 200e-3,
			'tau_o'     : 20e-3,
			'v_th'      : 0.61,
			'beta'      : 1.8
		}
		par['lif']['alpha'] = np.exp(-par['dt_sec']/par['lif']['tau_m']).astype(np.float64)
		par['lif']['rho']   = np.exp(-par['dt_sec']/par['lif']['tau_a']).astype(np.float64)
		par['lif']['kappa'] = np.exp(-par['dt_sec']/par['lif']['tau_o']).astype(np.float64)


update_dependencies()
load_custom_weights()
print('--> Parameters loaded.\n')
