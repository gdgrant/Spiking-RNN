from utils import *
from parameters import par, update_dependencies
from stimulus import Stimulus

class NetworkController:

	def __init__(self):
		""" Load initial network ensemble state """

		self.make_constants()
		self.make_variables()
		if par['use_adam']:
			self.make_adam_variables()

		self.size_ref = cp.ones([par['batch_size'],par['n_hidden']], \
			dtype=cp.float32)


	def make_variables(self):
		""" Pull network variables into GPU """

		if par['cell_type'] == 'rate':
			var_names = ['W_in', 'W_out', 'W_rnn', 'b_rnn', 'b_out', 'h_init']
		elif par['cell_type'] == 'adex':
			var_names = ['W_in', 'W_out', 'W_rnn']

		self.var_dict = {}
		for v in var_names:
			self.var_dict[v] = to_gpu(par[v+'_init'])


	def make_constants(self):
		""" Pull constants for computation into GPU """

		gen_constants   = ['n_hidden', 'W_rnn_mask', 'EI_mask', 'noise_rnn']
		time_constants  = ['alpha_neuron', 'beta_neuron', 'dt', 'num_time_steps']
		loss_constants  = ['loss_baseline', 'freq_cost', 'freq_target', 'reciprocal_cost', \
			'reciprocal_max', 'reciprocal_threshold']

		stp_constants   = ['syn_x_init', 'syn_u_init', 'U', 'alpha_stf', 'alpha_std', 'stp_mod']
		adex_constants  = ['adex', 'w_init']
		lat_constants   = ['latency_mask', 'max_latency']

		GA_constants    = ['mutation_rate', 'mutation_strength', 'cross_rate', 'loss_baseline', 'temperature']
		ES_constants    = ['ES_learning_rate', 'ES_sigma']

		local_learning_constants = ['local_learning_rate']

		constant_names  = gen_constants + time_constants + loss_constants
		constant_names += stp_constants if par['use_stp'] else []
		constant_names += adex_constants if par['cell_type'] == 'adex' else []
		constant_names += lat_constants if par['use_latency'] else []
		constant_names += GA_constants if par['learning_method'] in ['GA', 'TA'] else []
		constant_names += ES_constants if par['learning_method'] == 'ES' else []
		constant_names += local_learning_constants if par['local_learning'] else []

		self.con_dict = {}
		for c in constant_names:
			self.con_dict[c] = to_gpu(par[c])


	def make_adam_variables(self):
		""" Pull variables for managing ADAM into GPU """

		self.adam_par = {}
		self.adam_par['beta1']   = to_gpu(par['adam_beta1'])
		self.adam_par['beta2']   = to_gpu(par['adam_beta2'])
		self.adam_par['epsilon'] = to_gpu(par['adam_epsilon'])
		self.adam_par['t']       = to_gpu(0)

		for v in self.var_dict.keys():
			self.adam_par['m_' + v] = cp.zeros_like(self.var_dict[v][0])
			self.adam_par['v_' + v] = cp.zeros_like(self.var_dict[v][0])


	def update_constant(self, name, val):
		""" Update a given constant in the model """

		self.con_dict[name] = to_gpu(val)


	def load_batch(self, trial_info):
		""" Load a new batch of stimulus into the GPU """

		# Load the input data, target data, and mask to GPU
		self.input_data  = to_gpu(trial_info['neural_input'])
		self.output_data = to_gpu(trial_info['desired_output'])
		self.output_mask = to_gpu(trial_info['train_mask'])


	def run_models(self):
		""" Run network ensemble based on input data, collecting network
			outputs into y for later analysis and judgement """

		# Establish outputs and recording
		self.y = cp.zeros(par['y_init_shape'], dtype=cp.float32)
		self.spiking_means = cp.zeros([par['n_networks']])

		# Initialize cell states
		if par['cell_type'] == 'rate':
			spike = self.var_dict['h_init'] * self.size_ref
		elif par['cell_type'] == 'adex':
			spike = 0. * self.size_ref
			state = self.con_dict['adex']['V_r'] * self.size_ref
			adapt = self.con_dict['w_init'] * self.size_ref

		# Initialize STP if being used
		if par['use_stp']:
			syn_x = self.con_dict['syn_x_init'] * self.size_ref
			syn_u = self.con_dict['syn_u_init'] * self.size_ref
		else:
			syn_x = syn_u = 0.

		# Initialize latency buffer if being used
		if par['use_latency']:
			self.state_buffer = cp.zeros(par['state_buffer_shape'], dtype=cp.float32)

		# Set up derivative recording if using local_learning
		if par['local_learning']:
			self.local_learning(setup=True)

		# Apply the EI mask to the recurrent weights
		self.W_rnn_effective = apply_EI(self.var_dict['W_rnn'], self.con_dict['EI_mask'])

		# Loop across time and collect network output into y, using the
		# desired recurrent cell type
		for t in range(par['num_time_steps']):
			if par['cell_type'] == 'rate':
				spike, syn_x, syn_u = self.rate_recurrent_cell(spike, self.input_data[t], syn_x, syn_u, t)
				self.y[t,...] = cp.matmul(spike, self.var_dict['W_out']) + self.var_dict['b_out']
				self.spiking_means += cp.mean(spike, axis=(1,2))/self.con_dict['num_time_steps']

			elif par['cell_type'] == 'adex':
				spike, state, adapt = self.AdEx_recurrent_cell(spike, state, adapt, self.input_data[t], syn_x, syn_u, t)
				self.y[t,...] = (1-self.con_dict['beta_neuron'])*self.y[t-1,...] \
					+ self.con_dict['beta_neuron']*cp.matmul(spike, self.var_dict['W_out'])
				self.spiking_means += cp.mean(spike, axis=(1,2))*1000/self.con_dict['num_time_steps']

			if par['local_learning']:
				self.local_learning(t=t, spike=spike)


	def rnn_matmul(self, h_in, W_rnn, t):
		""" Perform the matmul operation required for the recurrent
			weight matrix, performing special operations such as latency
			where ncessary """

		if par['use_latency']:
			# Calculate this time step's latency-affected W_rnn and switch
			# to next time step
			W_rnn_latency = W_rnn[cp.newaxis,:,...] * self.con_dict['latency_mask'][:,cp.newaxis,...]
			self.con_dict['latency_mask'] = cp.roll(self.con_dict['latency_mask'], shift=1, axis=0)

			# Zero out the previous time step's buffer, and add to the
			# buffer for the upcoming time steps
			self.state_buffer[t-1%self.con_dict['max_latency'],...] = 0.
			self.state_buffer += cp.matmul(h_in, W_rnn_latency)

			# Return the hidden state buffer for this time step
			return self.state_buffer[t%self.con_dict['max_latency'],...]
		else:
			return cp.matmul(h_in, W_rnn)


	def rate_recurrent_cell(self, h, rnn_input, syn_x, syn_u, t):
		""" Process one time step of the hidden layer
			based on the previous state and the current input,
			using the rate-based model"""

		# Apply synaptic plasticity
		h_post, syn_x, syn_u = synaptic_plasticity(h, syn_x, syn_u, \
			self.con_dict, par['use_stp'], par['n_hidden'])

		# Calculate new hidden state
		h = relu((1-self.con_dict['alpha_neuron'])*h \
		  + self.con_dict['alpha_neuron']*(cp.matmul(rnn_input, self.var_dict['W_in']) \
		  + self.rnn_matmul(h_post, self.W_rnn_effective, t) + self.var_dict['b_rnn']) \
		  + cp.random.normal(scale=self.con_dict['noise_rnn'], size=h.shape).astype(cp.float32))

		return h, syn_x, syn_u


	def AdEx_recurrent_cell(self, spike, V, w, rnn_input, syn_x, syn_u, t):
		""" Process one time step of the hidden layer
			based on the previous state and the current input,
			using the adaptive-exponential spiking model """

		# Apply synaptic plasticity
		spike_post, syn_x, syn_u = synaptic_plasticity(spike, syn_x, syn_u, \
			self.con_dict, par['use_stp'], par['n_hidden'])

		# Calculate the current incident on the hidden neurons
		I = cp.matmul(rnn_input, self.var_dict['W_in']) + self.rnn_matmul(spike_post, self.W_rnn_effective)
		V, w, spike = run_adex(V, w, I, self.con_dict['adex'])

		return spike, V, w, syn_x, syn_u


	def judge_models(self):
		""" Determine the loss of each model, and rank them accordingly """

		# Calculate the task loss of each network (returns an array of size [n_networks])
		self.task_loss = cross_entropy(self.output_mask, self.output_data, self.y)


	def get_spiking(self):
		""" Return the spiking means of each network (unranked) """
		return to_cpu(self.spiking_means)


	def get_weights(self):
		""" Return the mean of the surviving networks' weights
			(post-sort, if sorted by the current learning method) """
		return to_cpu({name:np.mean(self.var_dict[name][:par['num_survivors'],...], axis=0) \
			for name in self.var_dict.keys()})


	def get_losses(self, ranked=True):
		""" Return the losses of each network, ranked if desired """
		if ranked:
			return to_cpu(self.loss[self.rank])
		else:
			return to_cpu(self.loss)


	def get_losses_by_type(self, ranked=True):
		""" Return the losses of each network, separated by type,
			and ranked if desired """
		if ranked:
			return to_cpu({'task':self.task_loss[self.rank], \
				'freq':self.freq_loss[self.rank], 'reci':self.reci_loss[self.rank]})
		else:
			return to_cpu({'task':self.task_loss, 'freq':self.freq_loss, 'reci':self.reci_loss})


	def get_performance(self, ranked=True):
		""" Return the accuracies of each network, ranked if desired """
		self.task_accuracy = accuracy(self.y, self.output_data, self.output_mask)
		self.full_accuracy = accuracy(self.y, self.output_data, self.output_mask, inc_fix=True)

		if ranked:
			return to_cpu(self.task_accuracy[self.rank]), to_cpu(self.full_accuracy[self.rank])
		else:
			return to_cpu(self.task_accuracy), to_cpu(self.full_accuracy)


def main():

	# Start the model run by loading the network controller and stimulus
	print('\nStarting model run: {}'.format(par['save_fn']))
	control = NetworkController()
	stim    = Stimulus()

	# Establish records for training loop
	save_record = {'iter':[], 'mean_task_acc':[], 'mean_full_acc':[], 'top_task_acc':[], \
		'top_full_acc':[], 'loss':[], 'mut_str':[], 'spiking':[], 'loss_factors':[]}

	t0 = time.time()
	# Run the training loop
	for i in range(par['iterations']):

		# Process a batch of stimulus using the current models
		control.load_batch(stim.make_batch())
		control.run_models()
		control.judge_models()
		task_accuracy, full_accuracy = control.get_performance()


if __name__ == '__main__':
	main()
