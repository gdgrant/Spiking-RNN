from imports import *
from model import main
from parameters import par, update_parameters

def print_parameters():

	keys = ['save_dir', 'savefn', 'cell_type', 'optimizer', \
		'batch_size', 'iterations', 'learning_rate', 'use_stp', \
		'n_input', 'n_hidden', 'n_output', \
		'train_input_weights', 'pseudo_th', 'dt', 'tau_hid', 'tau_out', \
		'latency', 'task', 'response_multiplier', 'dead_time', 'delay_time', \
		'var_delay', 'balance_EI_training']

	print('-'*60)
	for k in keys:
		print(k.ljust(30), par[k])
	print('-'*60)


delay = 120
pseudo_th = 40e-3
n = 5
r = int(sys.argv[1]) - 2
for j in range(n*r, n*(r+1)):

	savefn = 'impgrad_{:0>3}pth_{}neuron_var{}delay_v{:0>2}'.format(int(1000*par['pseudo_th']), par['n_hidden'], delay, j)

	updates = {
		'savefn'			: savefn,
		'pseudo_th'			: pseudo_th,
		'delay_time'		: delay,
		'iterations'		: 2000,
		'save_data_files'	: True }

	update_parameters(updates)
	print_parameters()
	main()