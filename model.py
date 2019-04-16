from utils import *
from parameters import par, update_dependencies
from adex import run_adex
from adaptive_lif import run_lif
from stimulus import Stimulus

class Model:

	def __init__(self):

		self.make_constants()
		self.make_variables()

		self.size_ref = cp.ones([par['batch_size'],par['n_hidden']])


	def make_variables(self):

		var_names = ['W_in', 'W_out', 'W_rnn', 'b_out']

		self.var_dict = {}
		for v in var_names:
			self.var_dict[v] = to_gpu(par[v+'_init'])


	def make_constants(self):

		constants = [
			'n_hidden', 'noise_rnn', 'adex', 'lif', \
			'w_init', 'beta_neuron', 'EI_mask']

		self.con_dict = {}
		for c in constants:
			self.con_dict[c] = to_gpu(par[c])


	def run_model(self, trial_info):

		# Load the input data, target data, and mask to GPU
		trial_info = to_gpu(trial_info)
		self.input_data  = trial_info['neural_input']
		self.output_data = trial_info['desired_output']
		self.output_mask = trial_info['train_mask']

		# Establish outputs
		self.y = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])

		# Initialize cell states
		if par['cell_type'] == 'lif':
			spike = 0. * self.size_ref
			state = 0. * self.size_ref
			adapt = 0. * self.size_ref
		elif par['cell_type'] == 'adex':
			spike = 0. * self.size_ref
			state = self.con_dict['adex']['V_r'] * self.size_ref
			adapt = self.con_dict['w_init'] * self.size_ref

		# Apply EI mask
		if par['EI_prop'] != 1.:
			self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])
		else:
			self.W_rnn_effective = self.var_dict['W_rnn']

		# Set cell type
		if par['cell_type'] == 'lif':
			cell = self.LIF_recurrent_cell
		elif par['cell_type'] == 'adex':
			cell = self.AdEx_recurrent_cell

		# Loop across time
		for t in range(par['num_time_steps']):

			spike, state, adapt, self.y[t,...] = \
				cell(self.input_data[t], spike, state, adapt, self.y[t-1,...])


	def LIF_recurrent_cell(self, rnn_input, z, v, a, y):

		I = rnn_input @ self.var_dict['W_in'] + z @ self.W_rnn_effective
		v, a, z = run_lif(v, a, I, self.con_dict['lif'])

		y = self.con_dict['lif']['kappa'] * y \
			+ z @ self.var_dict['W_out'] + self.var_dict['b_out']

		return z, v, a, y


	def AdEx_recurrent_cell(self, rnn_input, spike, V, w, y):

		I = rnn_input @ self.var_dict['W_in'] + spike @ self.W_rnn_effective
		V, w, spike = run_adex(V, w, I, self.con_dict['adex'])

		y = self.con_dict['beta_neuron'] * y \
			+ spike @ self.var_dict['W_out'] + self.var_dict['b_out']
		
		return spike, V, w, y


	def optimize(self):

		# Calculate task loss
		self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)


	def get_weights(self):

		return to_cpu({name:self.var_dict[name] for name in self.var_dict.keys()})


	def get_losses(self):

		return {'task':to_cpu(self.task_loss)}


	def get_performance(self):

		self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
		self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

		return to_cpu(self.task_accuracy), to_cpu(self.full_accuracy)


def main():

	# Start the model run by loading the network controller and stimulus
	print('\nStarting model run: {}'.format(par['save_fn']))
	model = Model()
	stim  = Stimulus()

	# Establish records for training loop
	save_record = {'iter':[], 'mean_task_acc':[], 'mean_full_acc':[], 'top_task_acc':[], \
		'top_full_acc':[], 'loss':[], 'mut_str':[], 'spiking':[], 'loss_factors':[]}

	t0 = time.time()
	# Run the training loop
	for i in range(par['iterations']):

		# Process a batch of stimulus using the current models
		model.run_model(stim.make_batch())
		model.optimize()
		task_accuracy, full_accuracy = model.get_performance()

		print(i, task_accuracy)


if __name__ == '__main__':
	main()
