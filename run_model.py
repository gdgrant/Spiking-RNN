from imports import *
from model import main
from parameters import par, update_dependencies

def print_parameters():

	keys = ['save_dir', 'savefn', 'cell_type', 'optimizer', \
		'batch_size', 'iterations', 'learning_rate', 'use_stp', \
		'n_input', 'n_hidden', 'n_output', 'gamma_psd', 'L_spike_cost', \
		'train_input_weights', 'pseudo_th', 'dt', 'tau_hid', 'tau_out', \
		'latency', 'task', 'response_multiplier', 'dead_time', 'delay_time']

	print('-'*60)
	for k in keys:
		print(k.ljust(30), par[k])
	print('-'*60)


print_parameters()
main()