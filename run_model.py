from imports import *
from model import main
from parameters import par, update_parameters, load_custom_weights

def print_parameters():

	keys = ['save_dir', 'savefn', 'load_weights', 'loadfn', 'spike_model', 'optimizer', \
		'batch_size', 'iterations', 'learning_rate', 'use_stp', \
		'n_input', 'n_hidden', 'n_output', 'dv_approx', 'betagrad', \
		'train_input_weights', 'pseudo_th', 'dt', 'tau_hid', 'tau_out', \
		'latency', 'task', 'response_multiplier', 'dead_time', 'delay_time', \
		'var_delay', 'balance_EI_training']

	print('-'*60)
	for k in keys:
		print(k.ljust(30), par[k])
	print('-'*60)

sweep       = False
load_single = True

if sweep:
	delay = 500
	dv_approx = True
	pseudoth = 10
	n = 5
	r = int(sys.argv[1])
	for j in range(n*r, n*(r+1)):

		savefn = 'taskswitch_fixedinp_no2nd_{}pseudoth_{}neuron_var{}delay_v{:0>2}'.format(pseudoth, par['n_hidden'], delay, j)

		updates = {
			'savefn'			: savefn,
			'task'				: 'dmswitch',
			'dv_approx'			: True,
			'betagrad'			: 0.,
			'psudo_th'			: pseudoth*1e-3,
			'delay_time'		: delay,
			'iterations'		: 10000,
			'save_data_files'	: True }

		update_parameters(updates)
		print_parameters()
		main()

if load_single:
	loadfn = './savedir/dmc_izhi_b25_wd6_fix_weights.pkl'
	new_par = pickle.load(open(loadfn, 'rb'))['par']
	update_parameters(new_par, verbose=False)
	update_parameters({'savefn':new_par['savefn'] + '_loaded', 'loadfn':loadfn, 'load_weights':True,})
	update_parameters({'balance_EI_training':False, 'batch_size':64})
	load_custom_weights()
	print_parameters()
	main()