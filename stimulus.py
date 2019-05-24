from imports import *
from parameters import par

class Stimulus:

	def __init__(self):

		self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


	def make_batch(self, var_delay=par['var_delay']):

		if par['task'] == 'dms':
			trial_info = self.dms()
		elif par['task'] == 'dmc':
			trial_info = self.dmc(var_delay)
		elif par['task'] == 'oic':
			trial_info = self.oic()
		else:
			raise Exception('Task "{}" not yet implemented.'.format(par['task']))

		if par['cell_type'] != 'rate':
			trial_info = self.make_spiking(trial_info)

		return trial_info


	def make_spiking(self, trial_info):

		do_plots = False

		if do_plots:
			import matplotlib.pyplot as plt
			fig, ax = plt.subplots(2,3)
			ax[0,0].set_xlabel('Neurons')
			ax[0,0].set_ylabel('Time')
			ax[0,0].set_title('Input')
			ax[0,1].set_title('Target')
			ax[0,2].set_title('Mask')

			ax[0,0].imshow(trial_info['neural_input'][:,0,:], aspect='auto', clim=[0,par['tuning_height']])
			ax[0,1].imshow(trial_info['desired_output'][:,0,:], aspect='auto', clim=[0,1])
			ax[0,2].imshow(trial_info['train_mask'][:,0,np.newaxis], aspect='auto', clim=[0,1])

		trial_info['neural_input'] = np.where(\
			trial_info['neural_input']/1000*par['dt'] > np.random.rand(*trial_info['neural_input'].shape), \
			np.ones_like(trial_info['neural_input']), np.zeros_like(trial_info['neural_input']))

		if do_plots:
			ax[1,0].imshow(trial_info['neural_input'][:,0,:], aspect='auto', clim=[0,1])
			ax[1,1].imshow(trial_info['desired_output'][:,0,:], aspect='auto', clim=[0,1])
			ax[1,2].imshow(trial_info['train_mask'][:,0,np.newaxis], aspect='auto', clim=[0,1])
			plt.show()

		return trial_info


	def dms(self):

		trial_info = {
			'neural_input'      : np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']]),
			'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']]),
			'train_mask'        : np.ones([par['num_time_steps'], par['batch_size']]),
		}

		end_dead_time       = par['dead_time']//par['dt']
		end_fix_time        = end_dead_time + par['fix_time']//par['dt']
		end_sample_time     = end_fix_time + par['sample_time']//par['dt']
		end_delay_time      = end_sample_time + par['delay_time']//par['dt']
		end_mask_time       = end_delay_time + par['mask_time']//par['dt']
		end_test_time       = end_delay_time + par['test_time']//par['dt']

		trial_info['timings'] = [end_delay_time]

		trial_info['train_mask'][:end_dead_time,...] = 0.
		trial_info['train_mask'][end_delay_time:end_mask_time,...] = 0.
		trial_info['train_mask'][end_mask_time:end_test_time,...] = par['response_multiplier']

		sample_direction = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])
		test_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])

		p_nonmatch = par['num_motion_dirs']/(2*par['num_motion_dirs']-2)    # <-- Provable
		match = np.random.choice([True, False], p=[1-p_nonmatch,p_nonmatch], size=par['batch_size'])
		test_direction = np.where(match, sample_direction, test_direction)
		match = np.where(test_direction==sample_direction, True, match)
		trial_info['match'] = match
		
		if par['fixation_on']:
			trial_info['neural_input'][:end_delay_time,:,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] += self.fix_tuning[np.newaxis,:,0]

		output_neuron = np.where(match, 1, 2)
		trial_info['desired_output'][:end_delay_time,:,0] = 1.
		trial_info['desired_output'][end_delay_time:end_test_time,np.arange(par['batch_size']),output_neuron] = 1.

		trial_info['neural_input'][end_fix_time:end_sample_time,np.arange(par['batch_size']),:par['num_motion_tuned']] \
			+= np.transpose(self.motion_tuning[np.newaxis,:,0,sample_direction[np.arange(par['batch_size'])]], [0,2,1])

		trial_info['neural_input'][end_delay_time:end_test_time,np.arange(par['batch_size']),:par['num_motion_tuned']] \
			+= np.transpose(self.motion_tuning[np.newaxis,:,0,test_direction], [0,2,1])

		return trial_info


	def dmc(self, var_delay=False):

		trial_info = {
			'neural_input'      : np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']]),
			'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']]),
			'train_mask'        : np.zeros([par['num_time_steps'], par['batch_size']]),
			'timings'           : np.zeros([2,par['batch_size']])
		}

		# Select match/nonmatch and catch trials
		sample_direction = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])
		test_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])
		if var_delay:
			catch_trials = np.random.choice([True,False], size=par['batch_size'], p=[par['catch_prob'], 1-par['catch_prob']])
		else:
			catch_trials = np.zeros([par['batch_size']], dtype=bool)

		sample_category  = sample_direction//int(par['num_motion_dirs']/2)
		test_category    = test_direction//int(par['num_motion_dirs']/2)

		match = sample_category == test_category
		trial_info['match']      = match
		trial_info['sample_cat'] = sample_category
		trial_info['test_cat']   = test_category
		trial_info['sample_dir'] = sample_direction
		trial_info['test_dir']   = test_direction

		output_neuron = np.where(match, 1, 2)

		# Fixed trial times
		end_dead_time       = par['dead_time']//par['dt']
		end_fix_time        = end_dead_time + par['fix_time']//par['dt']
		end_sample_time     = end_fix_time + par['sample_time']//par['dt']

		trial_info['timings'][0,:] = end_sample_time
		for t in range(par['batch_size']):

			if var_delay:
				delay = par['var_delay_max'] - int(np.random.exponential(scale=par['var_delay_max']/5))
				if delay < par['var_delay_min']:
					catch_trials[t] = True

				# delay = np.random.choice(par['delay_times'])//par['dt'] if var_delay else par['delay_time']//par['dt']
			else:
				delay = par['delay_time']//par['dt']

			# Variable trial times
			end_delay_time = end_sample_time + delay
			end_mask_time  = end_delay_time + par['mask_time']//par['dt']
			end_test_time  = par['num_time_steps'] - 1 #end_delay_time + par['test_time']//par['dt']

			# Save end of delay time unless catch trial
			trial_info['timings'][1,t] = end_delay_time

			# Generate sample stimulus
			trial_info['train_mask'][end_fix_time:end_delay_time,t] = 1.

			if par['fixation_on']:
				trial_info['neural_input'][:end_delay_time,t,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] += self.fix_tuning[np.newaxis,:,0]

			trial_info['desired_output'][:end_delay_time,t,0] = 1.
			trial_info['neural_input'][end_fix_time:end_sample_time,t,:par['num_motion_tuned']] \
				+= self.motion_tuning[:,0,sample_direction[t]]

			# Generate test stimulus (if not catch trial)
			if not catch_trials[t]:
				trial_info['train_mask'][end_mask_time:end_test_time,t] = par['response_multiplier']
				trial_info['desired_output'][end_delay_time:end_test_time,t,output_neuron[t]] = 1.
				trial_info['neural_input'][end_delay_time:end_test_time,t,:par['num_motion_tuned']] \
					+= self.motion_tuning[:,0,test_direction[t]]

		return trial_info


	def oic(self):

		trial_info = {
			'neural_input'      : np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']]),
			'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']]),
			'train_mask'        : np.ones([par['num_time_steps'], par['batch_size']])
		}

		end_dead_time       = par['dead_time']//par['dt']
		end_fix_time        = end_dead_time + par['fix_time']//par['dt']
		end_sample_time     = end_fix_time + par['sample_time']//par['dt']
		end_delay_time      = end_sample_time + par['delay_time']//par['dt']
		end_mask_time       = end_delay_time + par['mask_time']//par['dt']
		end_test_time       = end_delay_time + par['test_time']//par['dt']

		trial_info['timings'] = [end_sample_time, end_delay_time]

		trial_info['train_mask'][:end_dead_time,...] = 0.
		trial_info['train_mask'][end_delay_time:end_mask_time,...] = 0.
		trial_info['train_mask'][end_mask_time:end_test_time,...] = par['response_multiplier']

		sample_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])

		if par['fixation_on']:
			trial_info['neural_input'][:end_delay_time,:,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] += self.fix_tuning[np.newaxis,:,0]

		output_neuron = np.where(sample_direction < 4, 1, 2)
		trial_info['match'] = (output_neuron - 1).astype(np.bool)

		trial_info['desired_output'][end_dead_time:end_delay_time,:,0] = 1.
		trial_info['desired_output'][end_delay_time:end_test_time,np.arange(par['batch_size']),output_neuron] = 1.

		trial_info['neural_input'][end_fix_time:end_test_time,np.arange(par['batch_size']),:par['num_motion_tuned']] \
			+= np.transpose(self.motion_tuning[np.newaxis,:,0,sample_direction[np.arange(par['batch_size'])]], [0,2,1])

		return trial_info



	def create_tuning_functions(self):

		"""
		Generate tuning functions for the Postle task
		"""
		motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))
		fix_tuning    = np.zeros((par['num_fix_tuned'], par['num_receptive_fields']))
		rule_tuning   = np.zeros((par['num_rule_tuned'], par['num_rules']))

		# generate list of prefered directions
		# dividing neurons by 2 since two equal groups representing two modalities
		pref_dirs = np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields']))

		# generate list of possible stimulus directions
		stim_dirs = np.arange(0,360,360/par['num_motion_dirs'])

		for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
			for i in range(len(stim_dirs)):
				for r in range(par['num_receptive_fields']):
					d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
					n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
					motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

		for n in range(par['num_fix_tuned']):
			for i in range(par['num_receptive_fields']):
				if n%par['num_receptive_fields'] == i:
					fix_tuning[n,i] = par['tuning_height']

		for n in range(par['num_rule_tuned']):
			for i in range(par['num_rules']):
				if n%par['num_rules'] == i:
					rule_tuning[n,i] = par['tuning_height']


		return motion_tuning, fix_tuning, rule_tuning


if __name__ == '__main__':
	s = Stimulus()
	trial_info = s.make_batch(par['var_delay'])
