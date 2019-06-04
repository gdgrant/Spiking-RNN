# General imports and utility functions
from imports import *
from gpu_utils import *
from model_utils import *

# Training environment
from parameters import par, update_dependencies
from stimulus import Stimulus
from optimizers import Standard, AdamOpt
import plotting_functions as pf

import copy
import cupy.linalg as LA

# Network/cell model functions
from spike_models import run_spike_model
from dynamics_adex import calculate_dynamics as adex_dynamics
from dynamics_izhi import calculate_dynamics as izhi_dynamics


class Model:

	def __init__(self):

		self.init_constants()
		self.init_variables()
		self.init_optimizer()
		self.init_eligibility()

		self.size_ref = cp.ones([par['batch_size'], 1, par['n_hidden']])

		# Select model dynamics
		if par['spike_model'] == 'adex':
			self.dynamics = adex_dynamics
		elif par['spike_model'] == 'lif':
			self.dynamics = None	# Needs to be added
		elif par['spike_model'] == 'izhi':
			self.dynamics = izhi_dynamics


	def init_constants(self):
		""" Import constants from CPU to GPU """

		constants  = ['dt', 'dt_sec', 'adex', 'lif', 'izhi', 'w_init', 'v_init']
		constants += ['EI_vector', 'EI_matrix', 'EI_mask_exh', 'EI_mask_inh']
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

		self.grad_dict['W_rnn_exc_local']  = cp.zeros_like(self.var_dict['W_rnn'])
		self.grad_dict['W_rnn_inh_local']  = cp.zeros_like(self.var_dict['W_rnn'])
		self.grad_dict['W_in_local']  = cp.zeros_like(self.var_dict['W_in'])


	def init_optimizer(self):
		""" Initialize the optimizer to be used for this model """

		if par['optimizer']	== 'standard':
			self.optimizer = Standard(self.var_dict, par['learning_rate'])
		elif par['optimizer'] == 'adam':
			self.optimizer = AdamOpt(self.var_dict, par['learning_rate'], \
				par['adam_beta1'], par['adam_beta2'], par['adam_epsilon'])
		else:
			raise Exception('Optimizer "{}" not available.'.format(par['optimizer']))


	def init_eligibility(self):
		""" Make eligibility trace variables """

		self.eps = {}
		self.eps['inp'] = {}
		self.eps['rec'] = {}
		for s in ['v', 'w', 'ia']:
			self.eps['inp'][s] = cp.zeros([par['batch_size'], par['n_input'], par['n_hidden']])
		for s in ['v', 'w', 'ir', 'sx', 'su']:
			self.eps['rec'][s] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])

		self.kappa = {}
		self.kappa['inp'] = cp.zeros([par['batch_size'], par['n_input'], par['n_hidden']])
		self.kappa['rec'] = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])
		self.kappa['out'] = cp.zeros([par['batch_size'], par['n_hidden'], 1])


	def zero_state(self):
		""" Set all gradient and epsilon arrays to zero """
		""" Runs every iteration"""

		for v in self.var_names:
			self.grad_dict[v] = cp.zeros_like(self.grad_dict[v])

		for v in self.eps.keys():
			for s in self.eps[v].keys():
				self.eps[v][s] = cp.zeros_like(self.eps[v][s]) if not 'prev' in s else None

		self.eps['inp']['prev_v'] = [cp.zeros([par['batch_size'], par['n_input'], par['n_hidden']]) for _ in range(par['latency'])]
		self.eps['rec']['prev_v'] = [cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']]) for _ in range(par['latency'])]

		for k in self.kappa.keys():
			self.kappa[k] = cp.zeros_like(self.kappa[k])


	def apply_variable_rules(self):
		""" Apply rules to the variables that must be applied every
			time the model is run """

		self.eff_var = {}

		# Send input and output weights to effective variables
		self.eff_var['W_in']  = self.var_dict['W_in']
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


	def run_model(self, trial_info, testing=False):
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

		# Establish internal state recording
		self.v  = cp.zeros([par['num_time_steps'], par['batch_size'], 1, par['n_hidden']])
		self.w  = cp.zeros([par['num_time_steps'], par['batch_size'], 1, par['n_hidden']])
		self.sx = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden'], 1])
		self.su = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden'], 1])
		
		# Initialize cell states
		v = self.con_dict['v_init'] * self.size_ref
		w = self.con_dict['w_init'] * self.size_ref

		# Initialize synaptic plasticity
		sx = self.con_dict['syn_x_init'] if par['use_stp'] else 1.
		su = self.con_dict['syn_u_init'] if par['use_stp'] else 1.

		# Record other parts of the model as well
		self.z = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.h = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_hidden']])
		self.y = cp.zeros([par['num_time_steps'], par['batch_size'], par['n_output']])

		# Initialize input trace
		ia = cp.zeros([par['batch_size'], par['n_input'], par['n_hidden']])
		ir = cp.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']])

		self.I_sqr = 0

		# Make state dictionary
		state_dict = {'v':v, 'w':w, 'ia':ia, 'ir':ir, 'ja':copy.copy(ia), 'jr':copy.copy(ir), 'sx':sx, 'su':su}

		# Loop across time
		for t in range(par['num_time_steps']):

			# Run cell step
			state_dict, I = self.recurrent_cell(state_dict, t)

			self.I_sqr += (1/par['num_time_steps']) * cp.mean(cp.square(cp.sum(I, axis=1)))

			# Record cell state
			self.v[t,...]  = state_dict['v']
			self.w[t,...]  = state_dict['w']
			self.sx[t,...] = state_dict['sx']
			self.su[t,...] = state_dict['su']

			self.eps['inp']['prev_v'] = self.eps['inp']['prev_v'][1:]
			self.eps['rec']['prev_v'] = self.eps['rec']['prev_v'][1:]
			self.eps['inp']['prev_v'].append(self.eps['inp']['v'])
			self.eps['rec']['prev_v'].append(self.eps['rec']['v'])

			# Only run updates if training
			if not testing:

				# Update eligibilities and traces
				self.update_eligibility(state_dict, I, t)

				# Update pending weight changes
				self.calculate_weight_updates(t)


	def recurrent_cell(self, st, t):
		""" Compute one iteration of the recurrent network, progressing the
			internal state by one time step. """

		z = self.z[t-par['latency'],..., cp.newaxis]
		x = self.input_data[t,:,:,cp.newaxis]

		# Update the input traces based on presynaptic spikes
		curr_beta = self.con_dict[par['spike_model']]['beta']
		st['ia'] = curr_beta * st['ia'] + (1-curr_beta) * self.eff_var['W_in'] * x
		st['ir'] = curr_beta * st['ir'] + (1-curr_beta) * self.eff_var['W_rnn'] * st['sx'] * st['su'] * z

		st['ja'] = curr_beta * st['ja'] + (1-curr_beta) * x
		st['jr'] = curr_beta * st['jr'] + (1-curr_beta) * st['sx'] * st['su'] * z

		# Update the synaptic plasticity state (recurrent only; input is static)
		st['sx'], st['su'] = \
			synaptic_plasticity(st['sx'], st['su'], z, self.con_dict, par['use_stp'])

		# Sum the input currents into shape [batch x postsynaptic]
		I = cp.sum(st['ia'], axis=1, keepdims=True) + cp.sum(st['ir'], axis=1, keepdims=True)

		# Update the AdEx cell state with the input current
		st['v'], st['w'], self.z[t,...] = run_spike_model(st['v'], st['w'], I, par['spike_model'], self.con_dict[par['spike_model']])

		# Update output trace based on postsynaptic cell state (Eq. 12)
		self.y[t,...] = self.con_dict[par['spike_model']]['kappa'] * self.y[t-1,...] + self.z[t,...] @ self.eff_var['W_out'] + self.eff_var['b_out']

		# Calculate h, the pseudo-derivative (Eq. 5, ~24, 20/21)
		# Bellec et al., 2018b
		if par['spike_model'] == 'adex':
			self.h[t,...] = cp.squeeze(par['gamma_psd'] * cp.maximum(0., \
				1 - cp.abs((st['v']-(par['betagrad']+self.con_dict['adex']['V_T']))/par['pseudo_th'])))
		elif par['spike_model'] == 'izhi':
			self.h[t,...] = cp.squeeze(par['gamma_psd'] * cp.maximum(0., \
				1 - cp.abs((st['v']-(par['betagrad']+self.con_dict['izhi']['Vth']))/par['pseudo_th'])))
		else:
			raise Exception('Unimplemented pseudo-derivative.')

		#h = par['gamma_psd'] * cp.maximum(0., 1 - cp.abs((st['v'] + 40e-3)/par['pseudo_th']))
		#h = par['gamma_psd'] * cp.ones_like(h)
		return st, I


	def update_eligibility(self, state_dict, I, t):

		# Calculate the model dynamics and generate new epsilons
		self.eps = self.dynamics(self.eps, state_dict, self.input_data, self.z, self.h, \
			self.sx, self.su, self.con_dict, self.eff_var, self.var_dict, t)

		# Update and modulate e's

		e_inp = self.h[t,:,cp.newaxis,:] * self.eps['inp']['v']
		e_rec = self.h[t,:,cp.newaxis,:] * self.eps['rec']['v']
		e_out = self.z[t,...,cp.newaxis]

		# Increment kappa arrays forward in time (Eq. 42-45, k^(t-t') terms)
		self.kappa['inp'] = self.con_dict[par['spike_model']]['kappa']*self.kappa['inp'] + e_inp
		self.kappa['rec'] = self.con_dict[par['spike_model']]['kappa']*self.kappa['rec'] + e_rec
		self.kappa['out'] = self.con_dict[par['spike_model']]['kappa']*self.kappa['out'] + e_out

		# EI balance
		if par['balance_EI_training']:
			c = self.con_dict[par['spike_model']]
			h = np.squeeze(h, axis = 1)
			const = np.squeeze(c['dt'] * c['mu'] / c['C'], axis = (0,1))
			beta = 0.00002
			gamma = 0.000004

			self.grad_dict['W_rnn_exc_local'] += cp.mean((const * h * (1 - z))[:,np.newaxis,:] * state_dict['jr'], axis=0)
			self.grad_dict['W_rnn_exc_local'][:par['n_EI'],:] -= beta* cp.mean(self.eff_var['W_rnn'][:par['n_EI'],:], axis=0, keepdims=True)
			self.grad_dict['W_rnn_exc_local'][:par['n_EI'],:] -= gamma*self.eff_var['W_rnn'][:par['n_EI'],:]

			self.grad_dict['W_in_local'] += cp.mean((const * h * (1 - z))[:,np.newaxis,:] * state_dict['ja'], axis=0)
			self.grad_dict['W_in_local'] -= beta* cp.mean(self.eff_var['W_in'], axis=0, keepdims=True)
			self.grad_dict['W_in_local'] -= gamma* self.eff_var['W_in']

			self.grad_dict['W_rnn_inh_local'] =  cp.mean(I * state_dict['jr'], axis = 0)


	def calculate_weight_updates(self, t):

		# Calculate output error
		output_error = self.output_mask[t,:,cp.newaxis] * (self.output_data[t] - softmax(self.y[t]))

		L_hid = cp.sum(self.eff_var['W_out'][cp.newaxis,:,:] * output_error[:,cp.newaxis,:], axis=-1)
		L_out = output_error

		# Update pending weight changes
		if par['train_input_weights']:
			self.grad_dict['W_in'] += cp.mean(L_hid[:,cp.newaxis,:] * self.kappa['inp'], axis=0)
		self.grad_dict['W_rnn']    += cp.mean(L_hid[:,cp.newaxis,:] * self.kappa['rec'], axis=0)
		if par['balance_EI_training']:
			self.grad_dict['W_rnn']					+= self.con_dict['EI_mask_exh'] @ cp.mean(self.EI_balance_delta_exh, axis=0)
			self.grad_dict['W_rnn'][:par['n_EI'],:] -= self.EI_exh_limit
			self.grad_dict['W_rnn']					+= self.con_dict['EI_mask_inh'] @ cp.mean(self.EI_balance_delta_inh, axis=0)
		self.grad_dict['W_out']    += cp.mean(L_out[:,cp.newaxis,:] * self.kappa['out'], axis=0)
		self.grad_dict['b_out']    += cp.mean(L_out[:,cp.newaxis,:], axis=0)


	def optimize(self):
		""" Optimize the model -- apply any collected updates """

		if par['balance_EI_training']:

			self.grad_dict['W_rnn'] += 1e1*self.con_dict['EI_mask_exh'] @ self.grad_dict['W_rnn_exc_local']
			self.grad_dict['W_rnn'] += 1e1*self.con_dict['EI_mask_inh'] @ self.grad_dict['W_rnn_inh_local']
			self.grad_dict['W_in'] += 5e1*self.grad_dict['W_in_local']

			self.grad_dict['W_rnn_exc_local'] *= 0.
			self.grad_dict['W_rnn_inh_local'] *= 0.
			self.grad_dict['W_in_local'] *= 0.

		self.grad_dict['W_in'] *= self.con_dict['W_in_mask']
		self.grad_dict['W_rnn'] *= self.con_dict['W_rnn_mask']
		self.grad_dict['W_out'] *= self.con_dict['W_out_mask']

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


	def visualize_delta(self, i):

		return None
		#pf.visualize_delta(i, self.var_dict, self.grad_dict)


	def show_output_behavior(self, it, trial_info):

		pf.output_behavior(it, trial_info, softmax(self.y))


def main():

	# Start the model run by loading the network controller and stimulus
	print('\nLoading model...')
	model = Model()
	stim  = Stimulus()

	t0 = time.time()
	print('Starting training.\n')

	full_acc_record = []
	task_acc_record = []
	iter_record = []
	I_sqr_record = []
	W_rnn_grad_sum_record = []
	W_rnn_grad_norm_record = []

	# Run the training loop
	for i in range(par['iterations']):

		# Process a batch of stimulus using the current models
		trial_info = stim.make_batch()
		model.run_model(trial_info)
		model.optimize()

		losses = model.get_losses()
		mean_spiking = model.get_mean_spiking()
		task_accuracy, full_accuracy = model.get_performance()

		full_acc_record.append(full_accuracy)
		task_acc_record.append(task_accuracy)
		iter_record.append(i)
		I_sqr_record.append(model.I_sqr)
		W_rnn_grad_sum_record.append(cp.sum(model.grad_dict['W_rnn']))
		W_rnn_grad_norm_record.append(LA.norm(model.grad_dict['W_rnn']))

		info_str0 = 'Iter {:>5} | Task Loss: {:5.3f} | Task Acc: {:5.3f} | '.format(i, losses['task'], task_accuracy)
		info_str1 = 'Full Acc: {:5.3f} | Mean Spiking: {:6.3f} Hz'.format(full_accuracy, mean_spiking)
		print('Aggregating data...', end='\r')

		if par['plot_EI_testing']:
				# Plot I square
				plt.figure()
				plt.plot(I_sqr_record)
				plt.savefig('./savedir/{}_I_sqr_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
				plt.clf()
				plt.close()

				# Plot W_rnn sum update
				plt.figure()
				plt.plot(W_rnn_grad_sum_record)
				plt.savefig('./savedir/{}_W_rnn_grad_sum_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
				plt.clf()
				plt.close()

				# Plot W_rnn norm update
				plt.figure()
				plt.plot(W_rnn_grad_norm_record)
				plt.savefig('./savedir/{}_W_rnn_grad_norm_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
				plt.clf()
				plt.close()

		if i%50==0:
			pf.activity_plots(i, model)

			if i != 0:
				pf.training_curve(i, iter_record, full_acc_record, task_acc_record)

			if i%100 == 0:
				model.visualize_delta(i)

				if par['save_data_files']:
					data = {'par' : par, 'weights' : to_cpu(model.var_dict)}
					pickle.dump(data, open('./savedir/{}_data_iter{:0>6}.pkl'.format(par['savefn'], i), 'wb'))

			trial_info = stim.make_batch(var_delay=False)
			model.run_model(trial_info, testing=True)
			model.show_output_behavior(i, trial_info)

		# Print output info (after all saving of data is complete)
		print(info_str0 + info_str1)

		if i%100 == 0:
			if np.mean(task_acc_record[-100:]) > 0.9:
				print('\nMean accuracy greater than 0.9 over last 100 iters.\nMoving on to next model.\n')
				break


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		quit('\nQuit by KeyboardInterrupt.\n')
