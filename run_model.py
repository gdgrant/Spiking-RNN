from imports import *
from model import main
from parameters import par, update_parameters, load_custom_weights

def print_parameters():

	keys = ['save_dir', 'savefn', 'load_weights', 'loadfn', 'spike_model', 'optimizer', \
		'batch_size', 'iterations', 'learning_rate', 'use_stp', \
		'n_input', 'n_hidden', 'n_output', 'dv_approx', 'betagrad', \
		'train_input_weights', 'pseudo_th', 'dt', 'tau_hid', 'tau_out', \
		'latency', 'task', 'response_multiplier', 'dead_time', 'delay_time', \
		'var_delay', 'balance_EI_training', 'num_clusters', 'cluster_inh', 'cluster_conn_prob']

	print('-'*60)
	for k in keys:
		print(k.ljust(30), par[k])
	print('-'*60)

sweep       = True
load_single = False

if sweep:
	delay = 500
	dv_approx = True
	pseudoth = 20e-3
	betagrad = 10e-3

	num_clusters = 10
	predef_inp   = 0

	n_hidden = 250

	n = 5
	r = int(sys.argv[1])
	for j in range(n*r, n*(r+1)):

		# savefn = 'taskswitch_fixedinp_no2nd_{}pseudoth_{}neuron_var{}delay_v{:0>2}'.format(pseudoth, par['n_hidden'], delay, j)
		# savefn = 'fewer_clustered_secondorder_betagrad{:0>3}'.format(int(betagrad*1000))
		savefn = 'clopath_adLTDinv100_{}hidden_{}clusters_{}fixedinp'.format(n_hidden, num_clusters, predef_inp)

		updates = {
			'savefn'				: savefn,
			'task'					: 'dmc',
			'dv_approx'				: True,
			'betagrad'				: betagrad,
			'pseudo_th'				: pseudoth,
			'delay_time'			: delay,
			'n_hidden'				: n_hidden,
			'num_clusters'			: num_clusters,
			'train_input_weights'	: not bool(predef_inp),
			'iterations'			: 10000,
			'balance_EI_training'	: False,
			'save_data_files'		: True }

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