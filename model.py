from utils import *
from parameters import par, update_dependencies
from adex import run_adex
from adaptive_lif import run_lif
from stimulus import Stimulus
from optimizers import Standard, AdamOpt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model:

	def __init__(self):
		""" Initialize model with constants, variables, and references """

		self.init_constants()
		self.init_variables()
		self.init_eligibility()

		self.size_ref = cp.ones([par['batch_size'],par['n_hidden']])

		# For visualization of delta
		self.delta_W_out_hist = []
		self.delta_W_rnn_hist = []
		self.delta_W_inp_hist = []

		if par['optimizer']	== 'standard':
			self.optimizer = Standard(self.var_dict, par['learning_rate'])
		elif par['optimizer'] == 'adam':
			self.optimizer = AdamOpt(self.var_dict, par['learning_rate'], \
				par['adam_beta1'], par['adam_beta2'], par['adam_epsilon'])
		else:
			raise Exception('Optimizer "{}" not available.'.format(par['optimizer']))


	def init_constants(self):
		""" Import constants from CPU to GPU """

		constants = [
			'n_hidden', 'noise_rnn', 'adex', 'lif', \
			'w_init', 'beta_neuron', 'EI_mask', \
			'W_in_const', 'W_rnn_mask']

		self.con_dict = {}
		for c in constants:
			self.con_dict[c] = to_gpu(par[c])


	def init_variables(self):
		""" Import variables from CPU to GPU, and apply any one-time
			variable operations """

		self.var_names = ['W_in', 'W_out', 'W_rnn', 'b_out']

		self.var_dict  = {}
		self.grad_dict = {}
		for v in self.var_names:
			self.var_dict[v]  = to_gpu(par[v+'_init'])
			self.grad_dict[v] = cp.zeros_like(self.var_dict[v])


	def init_eligibility(self):
		""" Make eligibility trace variables """

		self.eps_v = {}
		self.eps_a = {}
		self.kappa = {}

		self.eps_v['inp'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_input']])
		self.eps_v['rec'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])

		self.eps_a['inp'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_input']])
		self.eps_a['rec'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])

		self.kappa['inp'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_input']])
		self.kappa['rec'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])
		self.kappa['out'] = cp.zeros([par['batch_size'], par['n_hidden']])


	def zero_state(self):
		""" Set all gradient and epsilon arrays to zero """

		for v in self.var_names:
			self.grad_dict[v] = cp.zeros_like(self.grad_dict[v])

		for v in self.eps_v.keys():
			self.eps_v[v] = cp.zeros_like(self.eps_v[v])
			self.eps_a[v] = cp.zeros_like(self.eps_a[v])

		for k in self.kappa.keys():
			self.kappa[k] = cp.zeros_like(self.kappa[k])
			

	def apply_variable_rules(self):
		""" Apply rules to the variables that must be applied every
			time the model is run """

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

		# Establish variable rules
		self.apply_variable_rules()
		
		# Clear gradients and epsilons
		self.zero_state()

		# Establish spike and output recording
		self.z = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.y = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])

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
			epsi = self.LIF_update_eligibility
		elif par['cell_type'] == 'adex':
			cell = self.AdEx_recurrent_cell
			epsi = self.AdEx_update_eligibility

		# Loop across time
		for t in range(par['num_time_steps']):

			# Get the spikes from par['latency'] time steps ago
			neuron_inds = np.arange(par['n_hidden']).astype(np.int64)
			latency_z = self.z[t-(1+par['latency_inds']),:,neuron_inds].T

			# Run cell step
			self.z[t,...], state, adapt, self.y[t,...], h = \
				cell(self.input_data[t], latency_z, state, adapt, self.y[t-1,...])

			quit()
			# Update eligibilities and traces
			epsi(self.input_data[t], self.z[t,...], h, t)

			# Update pending weight changes
			self.calculate_weight_updates(t)


	def LIF_update_eligibility(self, x, z, h, t):

		# Add dimension (separated for clarity)
		h = h[...,cp.newaxis]

		### Update epsilons
		# Calculate eligibility traces (Eq. 27)
		self.eps_a['inp'] = h * self.eps_v['inp'] + (par['lif']['rho'] - (h * par['lif']['beta'])) * self.eps_a['inp']
		self.eps_a['rec'] = h * self.eps_v['rec'] + (par['lif']['rho'] - (h * par['lif']['beta'])) * self.eps_a['rec']

		# Update trace of pre-synaptic activity [AKA x_hat and z_hat] (Eq. 5)
		self.eps_v['inp'] = par['lif']['alpha'] * self.eps_v['inp'] + x[:,cp.newaxis,:]
		self.eps_v['rec'] = par['lif']['alpha'] * self.eps_v['rec'] + z[:,:,cp.newaxis]

		### Update and modulate e's
		# Increment kappa arrays forward in time (Eq. 46-48, k^(t-t') terms)
		self.kappa['inp'] *= self.con_dict['lif']['kappa']
		self.kappa['rec'] *= self.con_dict['lif']['kappa']
		self.kappa['out'] *= self.con_dict['lif']['kappa']

		# Calculate trace impact on learning signal for input and recurrent weights (Eq. 5, 28, 47)
		self.kappa['inp'] += h * (self.eps_v['inp'] - par['lif']['beta'] * self.eps_a['inp'])
		self.kappa['rec'] += h * (self.eps_v['rec'] - par['lif']['beta'] * self.eps_a['rec'])

		# Calculate output impact on learning signal for output weights
		self.kappa['out'] += self.z[t,...]


	def AdEx_update_eligibility(self, x, z, h, t):

		pass


	def calculate_weight_updates(self, t):

		# Calculate output error
		output_error = self.output_mask[t,:,cp.newaxis] * (self.output_data[t] - softmax(self.y[t]))

		# Calculate learning signals per layer (Eq. 4)
		L_hid = cp.sum(self.var_dict['W_out'].T[cp.newaxis,...] * output_error[...,cp.newaxis], axis=1)
		L_out = output_error

		### Update pending weight changes
		self.grad_dict['W_in']  += cp.mean(L_hid[:,:,cp.newaxis] * self.kappa['inp'], axis=0).T
		self.grad_dict['W_rnn'] += cp.mean(L_hid[:,:,cp.newaxis] * self.kappa['rec'], axis=0).T
		self.grad_dict['W_out'] += cp.mean(L_out[:,:,cp.newaxis] * self.kappa['out'][:,cp.newaxis,:], axis=0).T
		self.grad_dict['b_out'] += cp.mean(L_out[:,:,cp.newaxis], axis=0).T


	def LIF_recurrent_cell(self, x, z, v, a, y):
		""" Run one step of the leaky-integrate-and-fire model
			x = input spikes, z = recurrent spikes, v = membrane voltage
			a = adaptation variable, y = previous output """

		# Calculate input current and get LIF cell states (Eq. 11, 25, 26)
		I = x @ self.var_dict['W_in'] + z @ self.W_rnn_effective
		v, a, z, A, v_th = run_lif(v, a, I, self.con_dict['lif'])

		# Calculate output based on current cell state (Eq. 12, 20, 21)
		y = self.con_dict['lif']['kappa'] * y + z @ self.var_dict['W_out'] + self.var_dict['b_out']

		# Calculate h, the pseudo-derivative (Eq. 5, ~24, 25/26)
		# Bellec et al., 2018b
		h = par['gamma'] * cp.maximum(0., 1 - cp.abs((v-A)/v_th))

		return z, v, a, y, h


	def AdEx_recurrent_cell(self, x, z, V, w, y):

		# Calculate input current and get AdEx cell states
		I = x @ self.var_dict['W_in'] + z @ self.W_rnn_effective
		V, w, z, v_th = run_adex(V, w, I, self.con_dict['adex'])

		# Calculate output based on current cell state (Eq. 12)
		y = self.con_dict['lif']['kappa'] * y + z @ self.var_dict['W_out'] + self.var_dict['b_out']

		# Calculate h, the pseudo-derivative (Eq. 5, ~24, 20/21)
		# Bellec et al., 2018b
		h = par['gamma'] * cp.maximum(0., 1 - cp.abs((V-v_th)/v_th))

		return z, V, w, y, h


	def optimize(self):
		""" Optimize the model -- apply any collected updates """

		# Calculate task loss
		self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)

		# Apply gradient updates using the chosen optimizer
		self.var_dict = self.optimizer.apply_gradients(self.grad_dict)

		# Saving delta W_out and delta W_rnn
		# self.delta_W_out_hist.append(cp.sum(self.delta_W_out))
		# self.delta_W_rnn_hist.append(cp.sum(self.delta_W_rnn))


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
		# fig, ax = plt.subplots(1,2, figsize=(12,8))
		# ax[0].plot(self.delta_W_rnn_hist)
		# ax[0].set_title('delta_W_rnn')
		# ax[1].plot(self.delta_W_out_hist)
		# ax[1].set_title('delta_W_out')

		# plt.savefig('./savedir/delta_iter{}.png'.format(i), bbox_inches='tight')
		# plt.clf()
		# plt.close()

		# # Imshow the delta W_out and W_rnn at iteration i

		# plt.figure()
		# plt.imshow(to_cpu(self.delta_W_rnn), aspect='auto')
		# plt.colorbar()
		# plt.savefig('./savedir/delta_W_rnn_iter{}.png'.format(i), bbox_inches='tight')
		# plt.close()

		# plt.figure()
		# plt.imshow(to_cpu(self.delta_W_out), aspect='auto')
		# plt.colorbar()
		# plt.savefig('./savedir/delta_W_out_iter{}.png'.format(i), bbox_inches='tight')
		# plt.close()
		pass


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


		if i%50 == 0:
			model.visualize_delta(i)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		quit('\nQuit by KeyboardInterrupt.\n')
