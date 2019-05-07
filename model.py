from utils import *
from parameters import par, update_dependencies
from adex import run_adex
from adaptive_lif import run_lif
from stimulus import Stimulus

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model:

	def __init__(self):
		""" Initialize model with constants, variables, and references """

		self.make_constants()
		self.make_variables()

		self.size_ref = cp.ones([par['batch_size'],par['n_hidden']])


	def make_constants(self):
		""" Import constants from CPU to GPU """

		constants = [
			'n_hidden', 'noise_rnn', 'adex', 'lif', \
			'w_init', 'beta_neuron', 'EI_mask', \
			'W_in_const', 'W_rnn_mask']

		self.con_dict = {}
		for c in constants:
			self.con_dict[c] = to_gpu(par[c])


	def make_variables(self):
		""" Import variables from CPU to GPU, and apply any one-time
			variable operations """

		var_names = ['W_in', 'W_out', 'W_rnn', 'b_out']

		self.var_dict = {}
		for v in var_names:
			self.var_dict[v] = to_gpu(par[v+'_init'])

		# Apply EI mask
		if par['EI_prop'] != 1.:
			self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])
		else:
			self.W_rnn_effective = self.var_dict['W_rnn']

		# Apply mask to the recurrent weights
		self.W_rnn_effective *= self.con_dict['W_rnn_mask']
		

	def run_model(self, trial_info):
		""" Run the model by:
			 - Loading trial data
			 - Setting initial states
			 - Iterating over trial data
			 - Collecting states and updates over time """

		# Load the input data, target data, and mask to GPU
		trial_info = to_gpu(trial_info)
		self.input_data  = trial_info['neural_input']
		self.output_data = trial_info['desired_output']
		self.output_mask = trial_info['train_mask']

		# Establish spike and output recording
		self.z = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.y = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])
		self.z_hat = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])

		epsilon_a = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])
		h         = cp.zeros([par['batch_size'], par['n_hidden']])

		#Optimization
		self.kappa_array_rnn = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])
		self.kappa_array_out = cp.zeros([par['batch_size'], par['n_hidden']])
		self.delta_W_rnn = cp.zeros([par['n_hidden'], par['n_hidden']])
		self.delta_W_out = cp.zeros([par['n_output'], par['n_hidden']])
		self.delta_b_out = cp.zeros([par['n_output'], 1])

		# Initialize cell states
		if par['cell_type'] == 'lif':
			state = 0. * self.size_ref
			adapt = 1. * self.size_ref
		elif par['cell_type'] == 'adex':
			state = self.con_dict['adex']['V_r'] * self.size_ref
			adapt = self.con_dict['w_init'] * self.size_ref

		# Set cell type
		if par['cell_type'] == 'lif':
			cell = self.LIF_recurrent_cell
		elif par['cell_type'] == 'adex':
			cell = self.AdEx_recurrent_cell

		# For visualization of delta
		self.delta_W_out_hist = []
		self.delta_W_rnn_hist = []

		# Loop across time
		for t in range(par['num_time_steps']):

			# Run cell step
			self.z[t,...], state, adapt, self.y[t,...], self.z_hat[t,...], epsilon_a, h = \
				cell(self.input_data[t], self.z[t-par['latency'],...], state, adapt, self.y[t-1,...], self.z_hat[t-1,...], epsilon_a)

			# Calculate output error
			output_error = self.output_data[t] - softmax(self.y[t])

			### Equations 4, 5, 46, 47
			# Calculate learning signal for recurrent weights
			L_rec = self.output_mask[t,:,cp.newaxis]*cp.sum(self.var_dict['W_out'].T[cp.newaxis,...] * output_error[...,cp.newaxis], axis=1)

			# Calculate trace impact on learning signal for rec. weights (Eq. 47)
			self.kappa_array_rnn *= self.con_dict['lif']['kappa']
			self.kappa_array_rnn += h[:,:,cp.newaxis] * (self.z_hat[t-1,...][:,cp.newaxis,:] - par['lif']['beta'] * epsilon_a)

			# Update recurrent weight delta
			self.delta_W_rnn += cp.mean(L_rec[:,:,cp.newaxis] * self.kappa_array_rnn, axis=0)

			### Equation 48
			# Calculate learning signal for output weights
			L_out = self.output_mask[t,:,cp.newaxis] * output_error

			# Calculate output impact on learning signal for output weights
			self.kappa_array_out *= self.con_dict['lif']['kappa']
			self.kappa_array_out += self.z[t,...]

			# Update output weight delta
			self.delta_W_out += cp.mean(L_out[:,:,cp.newaxis] @ self.kappa_array_out[:,cp.newaxis,:], axis=0)
			self.delta_b_out += cp.mean(L_out[:,:,cp.newaxis], axis=0)

	def LIF_recurrent_cell(self, rnn_input, z, v, a, y, z_hat, epsilon_a):

		# Calculate input current and get LIF cell states (Eq. 11, 25, 26)
		I = rnn_input @ self.var_dict['W_in'] + z @ self.W_rnn_effective
		v, a, z, A, v_th = run_lif(v, a, I, self.con_dict['lif'])

		# Calculate output based on current cell state (Eq. 12)
		y = self.con_dict['lif']['kappa'] * y + z @ self.var_dict['W_out'] + self.var_dict['b_out']

		# Update h, the pseudo-derivative (Eq. 5, ~24)
		# Bellec et al., 2018b
		h = par['gamma'] * cp.maximum(0., 1 - cp.abs((v-A)/v_th))

		# Calculate eligibility trace (Eq. 27)
		epsilon_a = h[:,:,cp.newaxis] @ z_hat[:,cp.newaxis,:] + (par['lif']['rho'] - (h[:,:,cp.newaxis] * par['lif']['beta'])) * epsilon_a

		# Update z_hat, the trace of pre-synaptic activity
		z_hat = par['lif']['alpha'] * z_hat + z

		return z, v, a, y, z_hat, epsilon_a, h


	def AdEx_recurrent_cell(self, rnn_input, z, V, w, y):

		I = rnn_input @ self.var_dict['W_in'] + z @ self.W_rnn_effective
		V, w, z = run_adex(V, w, I, self.con_dict['adex'])

		# Calculate output based on current cell state (Eq. 12)
		y = self.con_dict['lif']['kappa'] * y + z @ self.var_dict['W_out'] + self.var_dict['b_out']

		return z, V, w, y


	def optimize(self):
		""" Optimize the model -- apply any collected updates """

		# Calculate task loss
		self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)

		# Apply gradient updates
		self.var_dict['W_rnn'] += par['learning_rate'] * self.delta_W_rnn.T
		self.var_dict['W_out'] += par['learning_rate'] * self.delta_W_out.T
		self.var_dict['b_out'] += par['learning_rate'] * self.delta_b_out.T

		# Saving delta W_out and delta W_rnn
		self.delta_W_out_hist.append(cp.sum(self.delta_W_out))
		self.delta_W_rnn_hist.append(cp.sum(self.delta_W_rnn))

	def get_weights(self):

		return to_cpu({name:self.var_dict[name] for name in self.var_dict.keys()})


	def get_losses(self):

		return to_cpu({'task':self.task_loss})


	def get_mean_spiking(self):

		z_mean = cp.mean(self.z, axis=(1,2))
		spiking = cp.sum(z_mean*1000/par['trial_length'])

		return to_cpu(spiking)


	def get_performance(self):

		self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
		self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

		return to_cpu(self.task_accuracy), to_cpu(self.full_accuracy)


	def visualize_delta(self,i):
		# Plot the delta W_out and W_rnn over iterations
		fig, ax = plt.subplots(1,2, figsize=(12,8))
		ax[0].plot(self.delta_W_rnn_hist)
		ax[0].set_title('delta_W_rnn')
		ax[1].plot(self.delta_W_out_hist)
		ax[1].set_title('delta_W_out')

		plt.savefig('./savedir/delta_iter{}.png'.format(i), bbox_inches='tight')
		plt.clf()
		plt.close()

		# Imshow the delta W_out and W_rnn at iteration i

		plt.figure()
		plt.imshow(to_cpu(self.delta_W_rnn), aspect='auto')
		plt.colorbar()
		plt.savefig('./savedir/delta_W_rnn_iter{}.png'.format(i), bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.imshow(to_cpu(self.delta_W_out), aspect='auto')
		plt.colorbar()
		plt.savefig('./savedir/delta_W_out_iter{}.png'.format(i), bbox_inches='tight')
		plt.close()


def main():

	# Start the model run by loading the network controller and stimulus
	print('\nStarting model run: {}'.format(par['cell_type']))
	print('Task type: {}'.format(par['task']))

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

		losses = model.get_losses()
		mean_spiking = model.get_mean_spiking()
		task_accuracy, full_accuracy = model.get_performance()

		info_str0 = 'Iter {:>5} | Task Loss: {:5.3f} | Task Acc: {:5.3f} | '.format(\
			i, losses['task'], task_accuracy, full_accuracy)
		info_str1 = 'Full Acc: {:5.3f} | Mean Spiking: {:6.3f} Hz'.format(\
			full_accuracy, mean_spiking)
		print(info_str0 + info_str1)
		if i%10==0:
			fig, ax = plt.subplots(4,1, figsize=(16,10))
			ax[0].imshow(to_cpu(model.input_data[:,0,:].T), aspect='auto')
			ax[0].set_title('Input Data')
			ax[1].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
			ax[1].set_title('Spiking')
			ax[2].plot(to_cpu(np.mean(model.z[:,0,:], axis=(1))))
			ax[2].set_title('Trial 0 Mean Spiking')
			ax[3].plot(to_cpu(np.mean(model.z, axis=(1,2))))
			ax[3].set_title('All Trials Mean Spiking')


		if i%50 == 0:
			model.visualize_delta(i)

		fig, ax = plt.subplots(4,1, figsize=(16,10), sharex=True)
		ax[0].imshow(to_cpu(model.input_data[:,0,:].T), aspect='auto')
		ax[0].set_title('Input Data')
		ax[1].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
		ax[1].set_title('Spiking')
		ax[2].plot(to_cpu(np.mean(model.z[:,0,:], axis=(1))))
		ax[2].set_title('Trial 0 Mean Spiking')
		ax[2].set_xlim(0,par['num_time_steps'])
		ax[3].plot(to_cpu(np.mean(model.z, axis=(1,2))))
		ax[3].set_title('All Trials Mean Spiking')
		ax[3].set_xlim(0,par['num_time_steps'])

			plt.savefig('./savedir/diagnostic_iter{:0>4}.png'.format(i), bbox_inches='tight')
			plt.clf()
			plt.close()


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		quit('\nQuit by KeyboardInterrupt.\n')
