from imports import *
from gpu_utils import to_cpu
from sklearn.svm import SVC
from itertools import product

from parameters import par, update_parameters
from stimulus import Stimulus
from model import Model

# List save files to analyze
savefns = ['verify_500neuron_var120delay_v0_data_iter001900']

# Make plot
# fig, ax = plt.subplots(1, 2, figsize=(14,6))

# Iterate over provided save files
for num_fn, fn in enumerate(savefns):

	print('Processing file {} of {}.'.format(num_fn+1, len(savefns)))

	# Load data
	data = pickle.load(open('./savedir/{}.pkl'.format(fn), 'rb'))

	# Update parameters with current weights
	update_parameters(data['par'], verbose=False)
	update_parameters({'batch_size':128}, verbose=False)
	update_parameters({k+'_init':v for k,v in data['weights'].items()}, \
		verbose=False, update_deps=False)

	end_dead_time       = par['dead_time']//par['dt']
	end_fix_time        = end_dead_time + par['fix_time']//par['dt']
	end_sample_time     = end_fix_time + par['sample_time']//par['dt']
	end_delay_time      = end_sample_time + par['delay_time']//par['dt']

	# Make a new model and stimulus (which use the loaded parameters)
	print('\nLoading and running model.')
	model = Model()
	stim  = Stimulus()
	runs  = 8
	z_bin = 20//par['dt']

	c_all = []
	d_all = []
	z_all = []

	# Run a couple batches to generate sufficient data points
	for i in range(runs):
		print('R:{:>2}'.format(i), end='\r')
		trial_info = stim.make_batch(var_delay=False)
		model.run_model(trial_info, testing=True)

		c_all.append(trial_info['sample_cat'])
		d_all.append(trial_info['sample_dir'])
		z_all.append(to_cpu(model.z))

	del model
	del stim

	batch_size = runs*par['batch_size']

	c = np.concatenate(c_all, axis=0)
	d = np.concatenate(d_all, axis=0)
	z = np.concatenate(z_all, axis=1)

	print('Model run complete.')
	print('Performing ROC decoding on {} trials.\n'.format(batch_size))


	local_spikes = np.zeros([par['num_time_steps'], par['n_hidden']])
	for t in range(par['num_time_steps']-z_bin):
		fr = np.mean(np.sum(z[t:t+z_bin], axis=0)*1000/z_bin, axis=0)
		local_spikes[t] = fr

	plt.imshow(local_spikes, aspect='auto')
	plt.colorbar()
	plt.show()
	quit()
