# General imports and utility functions
from imports import *
from gpu_utils import to_gpu, to_cpu
from model_utils import *

# Training environment
from parameters import par, update_dependencies
from stimulus import Stimulus
from optimizers import Standard, AdamOpt

# Network/cell model functions
from adex import run_adex
from adex_dynamics import calculate_dynamics
from lif import run_lif


class Model:

	def __init__(self):

		self.init_constants()
		self.init_variables()
		self.init_optimizer()
		self.init_eligibility()

		self.size_ref = cp.ones([par['batch_size'], par['n_hidden']])


	def init_constants(self):
		""" Import constants from CPU to GPU """

		constants  = ['dt_sec', 'adex', 'lif', 'w_init']
		constants += ['EI_vector', 'EI_matrix']
		constants += ['W_in_mask', 'W_rnn_mask', 'W_out_mask', 'b_out_mask']

		if par['use_stp']:
			constants += ['alpha_stf', 'alpha_std', 'U', 'syn_x_init', 'syn_u_init']

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


	def init_optimizer(self):
		# Initialize the optimizer to be used for this model """

		if par['optimizer']	== 'standard':
			self.optimizer = Standard(self.var_dict, par['learning_rate'])
		elif par['optimizer'] == 'adam':
			self.optimizer = AdamOpt(self.var_dict, par['learning_rate'], \
				par['adam_beta1'], par['adam_beta2'], par['adam_epsilon'])
		else:
			raise Exception('Optimizer "{}" not available.'.format(par['optimizer']))


	def init_eligibility(self):
		""" Make eligibility trace variables """

		self.state_names = ['v', 'w', 'i', 'sx', 'su']

		self.eps = {}
		self.eps['inp'] = {}
		self.eps['rec'] = {}
		for s in self.state_names:
			self.eps['inp'][s] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_input']])
			self.eps['rec'][s] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])

		self.kappa = {}
		self.kappa['inp'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_input']])
		self.kappa['rec'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])
		self.kappa['out'] = cp.zeros([par['batch_size'], par['n_hidden']])


	def zero_state(self):
		""" Set all gradient and epsilon arrays to zero """

		for v in self.var_names:
			self.grad_dict[v] = cp.zeros_like(self.grad_dict[v])

		for v in self.eps.keys():
			for s in self.eps[v].keys():
				self.eps[v][s] = cp.zeros_like(self.eps[v][s])

		for k in self.kappa.keys():
			self.kappa[k] = cp.zeros_like(self.kappa[k])


	def apply_variable_rules(self):
		""" Apply rules to the variables that must be applied every
			time the model is run """

		self.eff_var = {}

		# Send input and output weights to effective variables
		self.eff_var['W_in'] = self.var_dict['W_in']
		self.eff_var['W_out'] = self.var_dict['W_out']
		self.eff_var['b_out'] = self.var_dict['b_out']

		# Send recurrent weights, with appropriate changes, to effective variables
		if par['EI_prop'] != 1.:
			self.eff_var['W_rnn'] = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_matrix'])
		else:
			self.eff_var['W_rnn'] = self.var_dict['W_rnn']

		# Apply masks for each weight, then divide by the current divider
		# to ensure correct conductance regime
		for k in self.eff_var.keys():
			self.eff_var[k] *= self.con_dict[k+'_mask']
			self.eff_var[k] /= par['current_divider']


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

		# Establish voltage, spike, and output recording
		self.v = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.z = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.y = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])

		# Initialize cell states
		if par['cell_type'] == 'lif':
			state = 0. * self.size_ref
			adapt = 1. * self.size_ref
			raise Exception('No cell or eligibility available for LIF.')

		elif par['cell_type'] == 'adex':
			state = self.con_dict['adex']['V_r'] * self.size_ref
			adapt = self.con_dict['w_init'] * self.size_ref

			cell  = self.AdEx_recurrent_cell
			epsi  = self.AdEx_update_eligibility

		# Initialize input trace
		trace = 0. * self.size_ref

		# Initialize synaptic plasticity
		syn_x = self.con_dict['syn_x_init'] if par['use_stp'] else 1.
		syn_u = self.con_dict['syn_u_init'] if par['use_stp'] else 1.

		# Make state dictionary
		state_dict = {'v':state, 'w':adapt, 'i':trace, 'sx':syn_x, 'su':syn_u}

		# Loop across time
		for t in range(par['num_time_steps']):

			# Get input spiking data
			x = self.input_data[t,...]
			
			# Get recurrent spikes from par['latency'] time steps ago
			neuron_inds = np.arange(par['n_hidden']).astype(np.int64)
			latency_z = self.z[t-(1+par['latency_inds']),:,neuron_inds].T

			# Run cell step
			self.z[t,...], self.y[t,...], h, state_dict = \
				cell(x, latency_z, self.y[t-1,...], state_dict)

			# Record membrane voltage
			self.v[t,...] = state_dict['v']

			# Update eligibilities and traces
			epsi(x, self.z[t,...], latency_z, state_dict, h, t)

			# Update pending weight changes
			self.calculate_weight_updates(t)


	def AdEx_recurrent_cell(self, x, z_i, y, st):

		# Calculate presynaptic activity
		presyn = x @ self.eff_var['W_in'] + z_i @ self.eff_var['W_rnn']

		# Update AdEx cell states (input current is 'i', the input trace)
		st['v'], st['w'], z_j = \
			run_adex(st['v'], st['w'], st['i'], self.con_dict['adex'])

		# Update input trace based on incoming spikes
		st['i'] = self.con_dict['adex']['beta'] * st['i'] \
			+ st['sx'] * st['su'] * presyn

		# Update synaptic plasticity
		st['sx'], st['su'] = synaptic_plasticity(st['sx'], st['su'], presyn, self.con_dict, par['use_stp'])

		# Update output trace based on postsynaptic cell state (Eq. 12)
		y = self.con_dict['adex']['kappa'] * y + z_j @ self.eff_var['W_out'] + self.eff_var['b_out']

		# Calculate h, the pseudo-derivative (Eq. 5, ~24, 20/21)
		# Bellec et al., 2018b
		h = par['gamma_psd'] * cp.maximum(0., 1 - cp.abs((st['v']-self.con_dict['adex']['V_T'])/par['pseudo_th']))

		return z_j, y, h, st


	def AdEx_update_eligibility(self, x, z, z_prev, state_dict, h, t):

		# Calculate the model dynamics and generate new epsilons
		self.eps = calculate_dynamics(self.eps, x, z, z_prev, state_dict, h, \
			self.con_dict, self.eff_var)

		# Update and modulate e's
		e_inp = h[...,cp.newaxis] * self.eps['inp']['v']
		e_rec = h[...,cp.newaxis] * self.eps['rec']['v']
		e_out = z

		# Increment kappa arrays forward in time (Eq. 42-45, k^(t-t') terms)
		self.kappa['inp'] = self.con_dict['adex']['kappa']*self.kappa['inp'] + e_inp
		self.kappa['rec'] = self.con_dict['adex']['kappa']*self.kappa['rec'] + e_rec
		self.kappa['out'] = self.con_dict['adex']['kappa']*self.kappa['out'] + e_out

	def calculate_weight_updates(self, t):

		# Calculate output error
		output_error = self.output_mask[t,:,cp.newaxis] * (self.output_data[t] - softmax(self.y[t]))

		# Calculate learning signals per layer (Eq. 4)
		L_hid = cp.sum(self.eff_var['W_out'].T[cp.newaxis,...] * output_error[...,cp.newaxis], axis=1) \
			- par['L_spike_cost']*cp.mean(self.z[t], axis=[1], keepdims=True)
		L_out = output_error

		### Update pending weight changes
		if par['train_input_weights']:
			self.grad_dict['W_in']  += cp.mean(L_hid[:,:,cp.newaxis] * self.kappa['inp'], axis=0).T
		self.grad_dict['W_rnn'] += cp.mean(L_hid[:,:,cp.newaxis] * self.kappa['rec'], axis=0).T
		self.grad_dict['W_out'] += cp.mean(L_out[:,:,cp.newaxis] * self.kappa['out'][:,cp.newaxis,:], axis=0).T
		self.grad_dict['b_out'] += cp.mean(L_out[:,:,cp.newaxis], axis=0).T


	def optimize(self):
		""" Optimize the model -- apply any collected updates """

		# Calculate task loss
		self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)

		# Apply gradient updates using the chosen optimizer
		self.var_dict = self.optimizer.apply_gradients(self.grad_dict)


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

		for n in self.grad_dict.keys():
			fig, ax = plt.subplots(1,2, figsize=[16,8])
			im = ax[0].imshow(to_cpu(par['learning_rate']*self.grad_dict[n]), aspect='auto')
			fig.colorbar(im, ax=ax[0])
			im = ax[1].imshow(to_cpu(self.var_dict[n]), aspect='auto')
			fig.colorbar(im, ax=ax[1])

			fig.suptitle(n)
			ax[0].set_title('Gradient')
			ax[1].set_title('Variable')
			
			plt.savefig('./savedir/{}_delta_{}_iter{}.png'.format(par['savefn'], n, i), bbox_inches='tight')
			plt.clf()
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

		V_min = to_cpu(model.v[:,0,:].T.min())

		if i%10==0:
			fig, ax = plt.subplots(4,1, figsize=(15,11), sharex=True)
			ax[0].imshow(to_cpu(model.input_data[:,0,:].T), aspect='auto')
			ax[0].set_title('Input Data')
			ax[1].imshow(to_cpu((model.input_data[:,0,:] @ model.var_dict['W_in']).T), aspect='auto')
			ax[1].set_title('Projected Inputs')
			ax[2].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
			ax[2].set_title('Spiking')
			ax[3].imshow(to_cpu(model.v[:,0,:].T), aspect='auto', clim=(V_min,0.))
			ax[3].set_title('Membrane Voltage ($(V_r = {:5.3f}), {:5.3f} \\leq V_j^t \\leq 0$)'.format(par['adex']['V_r'].min(), V_min))

			ax[0].set_ylabel('Input Neuron')
			ax[1].set_ylabel('Hidden Neuron')
			ax[2].set_ylabel('Hidden Neuron')
			ax[3].set_ylabel('Hidden Neuron')

			plt.savefig('./savedir/{}_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
			plt.clf()
			plt.close()


		if i%50 == 0:
			model.visualize_delta(i)


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		quit('\nQuit by KeyboardInterrupt.\n')
