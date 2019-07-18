from imports import *
from model import main
from parameters import par, update_parameters, load_custom_weights

def print_parameters():

	keys = ['save_dir', 'savefn', 'load_weights', 'loadfn', 'spike_model', 'optimizer', \
		'batch_size', 'iterations', 'learning_rate', 'use_stp', \
		'n_input', 'n_hidden', 'n_output',  'betagrad', \
		'train_input_weights', 'pseudo_th', 'dt', 'tau_hid', 'tau_out', \
		'latency', 'task', 'response_multiplier', 'dead_time', 'delay_time', \
		'var_delay', 'num_clusters', 'cluster_inh', \
		'cluster_conn_prob', 'defined_input_weights']

	print('-'*60)
	for k in keys:
		print(k.ljust(30), par[k])
	print('-'*60)


delay = 0
pseudoth = 20e-3 # used for psuedo-derivative
betagrad = 10e-3 # used for psuedo-derivative


# savefn = 'taskswitch_fixedinp_no2nd_{}pseudoth_{}neuron_var{}delay_v{:0>2}'.format(pseudoth, par['n_hidden'], delay, j)
# savefn = 'fewer_clustered_secondorder_betagrad{:0>3}'.format(int(betagrad*1000))
savefn = 'oic_n250_beta10'

updates = {
	'savefn'				: savefn,
	'task'					: 'oic',
	'n_hidden'				: 250,
	'betagrad'				: betagrad,
	'pseudo_th'				: pseudoth,
	'delay_time'			: delay,
	'iterations'			: 10000,
	'save_data_files'		: True }

update_parameters(updates)
print_parameters()
main()
