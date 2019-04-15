from parameters import *

class Stimulus:

    def __init__(self):

        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


    def make_batch(self):

        if par['task'] == 'dms':
            trial_info = self.dms()
        elif par['task'] == 'oic':
            trial_info = self.oic()
        else:
            raise Exception('Task "{}" not yet implemented.'.format(par['task']))

        if par['cell_type'] != 'rate':
            trial_info = self.make_spiking(trial_info)

        trial_info['neural_input']   = trial_info['neural_input'][:,np.newaxis,...]
        trial_info['desired_output'] = trial_info['desired_output'][:,np.newaxis,...]
        trial_info['train_mask']     = trial_info['train_mask'][:,np.newaxis,...]

        return trial_info


    def make_spiking(self, trial_info):

        do_plots = False

        if do_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,3)
            ax[0,0].imshow(trial_info['neural_input'][:,0,:].astype(np.float32), aspect='auto', clim=[0,par['tuning_height']])
            ax[0,1].imshow(trial_info['desired_output'][:,0,:].astype(np.float32), aspect='auto', clim=[0,1])
            ax[0,2].imshow(trial_info['train_mask'][:,0,np.newaxis].astype(np.float32), aspect='auto', clim=[0,1])

            ax[0,0].set_xlabel('Neurons')
            ax[0,0].set_ylabel('Time')
            ax[0,0].set_title('Input')
            ax[0,1].set_title('Target')
            ax[0,2].set_title('Mask')

        trial_info['neural_input'] = np.where(trial_info['neural_input']/1000*par['dt'] > np.random.rand(*trial_info['neural_input'].shape), \
            np.ones_like(trial_info['neural_input']), np.zeros_like(trial_info['neural_input']))

        if do_plots:
            ax[1,0].imshow(trial_info['neural_input'][:,0,:].astype(np.float32), aspect='auto', clim=[0,1])
            ax[1,1].imshow(trial_info['desired_output'][:,0,:].astype(np.float32), aspect='auto', clim=[0,1])
            ax[1,2].imshow(trial_info['train_mask'][:,0,np.newaxis].astype(np.float32), aspect='auto', clim=[0,1])
            plt.show()
            quit()

        return trial_info


    def dms(self):

        trial_info = {
            'neural_input'      : np.float32(np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']])),
            'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']], dtype=np.float32),
            'train_mask'        : np.ones([par['num_time_steps'], par['batch_size']], dtype=np.float32)
        }

        end_dead_time       = par['dead_time']//par['dt']
        end_fix_time        = end_dead_time + par['fix_time']//par['dt']
        end_sample_time     = end_fix_time + par['sample_time']//par['dt']
        end_delay_time      = end_sample_time + par['delay_time']//par['dt']
        end_mask_time       = end_delay_time + par['mask_time']//par['dt']
        end_test_time       = end_delay_time + par['test_time']//par['dt']

        trial_info['train_mask'][:end_dead_time,...] = 0.
        trial_info['train_mask'][end_delay_time:end_mask_time,...] = 0.
        trial_info['train_mask'][end_mask_time:end_test_time,...] = par['response_multiplier']

        match = np.random.choice([True, False], size=par['batch_size'])
        sample_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])

        for t in range(par['batch_size']):

            test_direction = sample_direction[t] if match[t] else np.random.choice(np.setdiff1d(np.arange(par['num_motion_dirs']), sample_direction[t]))

            if par['fixation_on']:
                trial_info['neural_input'][end_dead_time:end_delay_time,t,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] += self.fix_tuning[np.newaxis,:,0]
            trial_info['neural_input'][end_fix_time:end_sample_time,t,:par['num_motion_tuned']] += self.motion_tuning[np.newaxis,:,0,sample_direction[t]]
            trial_info['neural_input'][end_delay_time:end_test_time,t,:par['num_motion_tuned']] += self.motion_tuning[np.newaxis,:,0,test_direction]

            output_neuron = 1 if match[t] else 2
            trial_info['desired_output'][:end_delay_time,t,0] = 1.
            trial_info['desired_output'][end_delay_time:end_test_time,t,output_neuron] = 1.

        return trial_info


    def oic(self):

        trial_info = {
            'neural_input'      : np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']]),
            'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']], dtype=np.float32),
            'train_mask'        : np.ones([par['num_time_steps'], par['batch_size']], dtype=np.float32)
        }

        end_dead_time       = par['dead_time']//par['dt']
        end_fix_time        = end_dead_time + par['fix_time']//par['dt']
        end_sample_time     = end_fix_time + par['sample_time']//par['dt']
        end_delay_time      = end_sample_time + par['delay_time']//par['dt']
        end_mask_time       = end_delay_time + par['mask_time']//par['dt']
        end_test_time       = end_delay_time + par['test_time']//par['dt']

        trial_info['train_mask'][:end_dead_time,...] = 0.
        trial_info['train_mask'][end_delay_time:end_mask_time,...] = 0.
        trial_info['train_mask'][end_mask_time:end_test_time,...] = par['response_multiplier']

        sample_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])

        for t in range(par['batch_size']):

            if par['fixation_on']:
                trial_info['neural_input'][end_dead_time:end_delay_time,t,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] += self.fix_tuning[np.newaxis,:,0]
            trial_info['neural_input'][end_fix_time:end_test_time,t,:par['num_motion_tuned']] += self.motion_tuning[np.newaxis,:,0,sample_direction[t]]

            output_neuron = 1 if sample_direction[t] < 4 else 2
            trial_info['desired_output'][end_dead_time:end_delay_time,t,0] = 1.
            trial_info['desired_output'][end_delay_time:end_test_time,t,output_neuron] = 1.

        return trial_info



    def create_tuning_functions(self):

        """
        Generate tuning functions for the Postle task
        """
        motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']), dtype=np.float32)
        fix_tuning    = np.zeros((par['num_fix_tuned'], par['num_receptive_fields']), dtype=np.float32)
        rule_tuning   = np.zeros((par['num_rule_tuned'], par['num_rules']), dtype=np.float32)

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

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
    s.make_batch()
