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

        self.size_ref = cp.ones([par['n_networks'],par['batch_size'],par['n_hidden']], \
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

        gen_constants   = ['n_networks', 'n_hidden', 'W_rnn_mask', 'EI_mask', 'noise_rnn']
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

        # Calculate the frequency loss of each network (returns an array of size [n_networks])
        self.freq_loss = self.con_dict['freq_cost']*cp.abs(self.spiking_means-self.con_dict['freq_target'])

        # Calculate the reciprocal weights loss of each network (returns an array of size [n_networks])
        weight_ref = self.var_dict['W_rnn'][:,:par['n_EI'],:par['n_EI']] > self.con_dict['reciprocal_threshold']
        self.reci_loss = cp.mean(weight_ref * weight_ref.transpose([0,2,1]), axis=(1,2))
        self.reci_loss = -self.con_dict['reciprocal_cost']*cp.minimum(self.con_dict['reciprocal_max'], self.reci_loss)

        # Aggregate the various loss terms
        self.loss = self.task_loss + self.freq_loss + self.reci_loss

        # If a network explodes due to a poorly-selected recurrent connection,
        # set that network's loss to the loss baseline (chosen prior to the 0th iteration)
        #self.loss[cp.where(cp.isnan(self.loss))] = self.con_dict['loss_baseline']

        # Rank the networks (returns [n_networks] indices)
        self.rank = cp.argsort(self.loss.astype(cp.float32)).astype(cp.int16)

        # Sort the weights if required by the current learning method
        if par['learning_method'] in ['GA', 'TA']:
            # TODO: should we rank self.loss here for consistency?
            for name in self.var_dict.keys():
                self.var_dict[name] = self.var_dict[name][self.rank,...]
                if par['local_learning'] and name in par['local_learning_vars']:
                    self.local_delta[name] = self.local_delta[name][self.rank,...]


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


    def local_learning(self, t=None, spike=None, setup=False):
        """ Process a step of the local learning algorithm
            (or set up the local learning environment) """

        if setup:
            self.local_delta = {}
            self.local_delta['W_out'] = cp.zeros(self.var_dict['W_out'].shape, dtype=cp.float32)
            if par['cell_type'] == 'rate':
                self.local_delta['b_out'] = cp.zeros(self.var_dict['b_out'].shape, dtype=cp.float32)
        else:
            if t is not None and spike is not None:
                delta = self.output_mask[t,...,cp.newaxis] * (self.output_data[t,...] - softmax(self.y[t,...]))
                for k in range(par['n_output']):
                    self.local_delta['W_out'][...,k] += cp.mean(delta[...,k:k+1]*spike, axis=1)/self.con_dict['num_time_steps']
                if par['cell_type'] == 'rate':
                    self.local_delta['b_out'] += cp.mean(delta, axis=1, keepdims=True)/self.con_dict['num_time_steps']
            else:
                raise Exception('Processing local learning requires a valid time step and spike state.')


    def breed_models_genetic(self):
        """ Based on the first s networks in the ensemble, produce more networks
            slightly mutated from those s """

        for s, name in itertools.product(range(par['num_survivors']), self.var_dict.keys()):
            indices = cp.arange(s+par['num_survivors'], par['n_networks'], par['num_survivors'])

            if par['use_crossing']:
                raise Exception('Crossing not currently implemented for TA.')

            if par['local_learning']:
                raise Exception('LL not currently implemented for GA.')

            self.var_dict[name][indices,...] = mutate(self.var_dict[name][s,...], indices.shape[0], \
                self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


    def breed_models_thermal(self, iteration):
        """ Based on the top networks in the ensemble, probabilistically
            produce more networks slightly mutated from those top networks,
            selecting the sampled networks using simulated annealing """

        if par['use_crossing']:
            raise Exception('Crossing not currently implemented for TA.')
        if par['local_learning']:
            # perform local learning every even iteration, and thermal annealing every odd iteration
            for name in par['local_learning_vars']:
                self.var_dict[name] += self.con_dict['local_learning_rate'] * self.local_delta[name]

        corrected_loss = self.loss[self.rank]
        corrected_loss[cp.where(cp.isnan(self.loss))] = 999.
        prob_of_return = softmax(-corrected_loss/self.con_dict['temperature'])
        # TODO: set replace=False but make sure we have par['num_survivors'] amount of samples with non-zero prob
        samples = np.random.choice(par['n_networks'], size=[par['num_survivors']], p=to_cpu(prob_of_return), replace=True)
        num_mutations = (par['n_networks']-par['num_survivors'])//par['num_survivors']

        #uniques = list(set(samples.tolist()))
        #print('\nAnnealing diagnostics:\nNum. unique sampled networks:' +
        #    ' {}\nWorst sampled network ID:     {}\n'.format(len(uniques), sorted(uniques)[-1]))

        for name in self.var_dict.keys():
            self.var_dict[name][:par['num_survivors']] = self.var_dict[name][samples]
            for i in range(par['num_survivors']):
                #mutation_subset = range(par['num_survivors']+i*num_mutations, \
                #    par['num_survivors']+(i+1)*num_mutations)
                self.var_dict[name][par['num_survivors']+i*num_mutations:par['num_survivors']+(i+1)*num_mutations,...] \
                    = mutate(self.var_dict[name][i,...], num_mutations, \
                    self.con_dict['mutation_rate'], self.con_dict['mutation_strength'])

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


    def breed_models_evo_search(self, iteration):
        """ Using the 0th model in the ensemble as the base network, calculate
            the gradient of the loss from the previous run and adjust the base
            network parameter accordingly, using the evolutionary search
            algorithm.  Adam and local learning are invoked as necessary. """

        if par['use_adam']:
            self.adam_par['t'] += 1
            learning_rate = self.con_dict['ES_learning_rate'] * \
                cp.sqrt(1-self.adam_par['beta2']**self.adam_par['t'])/(1-self.adam_par['beta1']**self.adam_par['t'])
        else:
            learning_rate = self.con_dict['ES_learning_rate']

        for name in self.var_dict.keys():
            if iteration == 0:
                self.var_dict[name] = self.var_dict[name][self.rank,...]
            else:
                if par['local_learning'] and name in par['local_learning_vars']:
                    delta_var = self.local_delta[name][0]/self.con_dict['num_time_steps']
                else:
                    grad_epsilon = self.var_dict[name][1:,...] - self.var_dict[name][0:1,...]
                    delta_var = -cp.mean(grad_epsilon * self.loss[1:,cp.newaxis,cp.newaxis], axis=0)/cp.std(self.loss[1:])

                if par['use_adam']:
                    self.adam_par['m_' + name] = self.adam_par['beta1']*self.adam_par['m_' + name] + \
                        (1 - self.adam_par['beta1'])*delta_var
                    self.adam_par['v_' + name] = self.adam_par['beta2']*self.adam_par['v_' + name] + \
                        (1 - self.adam_par['beta2'])*delta_var*delta_var
                    self.var_dict[name][0] += learning_rate * self.adam_par['m_' + name]/(self.adam_par['epsilon'] + \
                        cp.sqrt(self.adam_par['v_' + name]))
                else:
                    self.var_dict[name][0] += learning_rate * delta_var

                if not (par['local_learning'] and name in par['local_learning_vars']):
                    var_epsilon = cp.random.normal(0, self.con_dict['ES_sigma'], \
                        size=self.var_dict[name][1::2,...].shape).astype(cp.float32)
                    var_epsilon = cp.concatenate([var_epsilon, -var_epsilon], axis=0)
                    self.var_dict[name][1:,...] = self.var_dict[name][0:1,...] + var_epsilon
                else:
                    self.var_dict[name][1:,...] = self.var_dict[name][0:1,...]

        self.var_dict['W_rnn'] *= self.con_dict['W_rnn_mask']


def main():

    # Start the model run by loading the network controller and stimulus
    print('\nStarting model run: {}'.format(par['save_fn']))
    control = NetworkController()
    stim    = Stimulus()

    # Select whether to get losses ranked, according to learning method
    if par['learning_method'] in ['GA', 'TA']:
        is_ranked = True
    elif par['learning_method'] == 'ES':
        is_ranked = False
    else:
        raise Exception('Unknown learning method: {}'.format(par['learning_method']))

    # Get loss baseline and update the ensemble reference accordingly
    control.load_batch(stim.make_batch())
    control.run_models()
    control.judge_models()
    loss_baseline = np.nanmean(control.get_losses(is_ranked))
    control.update_constant('loss_baseline', loss_baseline)

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

        # Get the current loss scores
        loss = control.get_losses(is_ranked)

        # Apply optimizations based on the current learning method(s)
        mutation_strength = 0.
        if par['learning_method'] in ['GA', 'TA']:
            mutation_strength = par['mutation_strength']*(np.nanmean(loss[:par['num_survivors']])/loss_baseline)
            control.update_constant('mutation_strength', mutation_strength)
            thresholds = [0.25, 0.1, 0.05, 0]
            modifiers  = [1/2, 1/4, 1/8]
            for t in range(len(thresholds))[:-1]:
                if thresholds[t] > mutation_strength > thresholds[t+1]:
                    mutation_strength = par['mutation_strength']*np.nanmean(loss)/loss_baseline * modifiers[t]
                    break

            if par['learning_method'] == 'GA':
                control.breed_models_genetic()
            elif par['learning_method'] == 'TA':
                control.update_constant('temperature', par['temperature']*par['temperature_decay']**i)
                control.breed_models_thermal(i)

        elif par['learning_method'] == 'ES':
            control.breed_models_evo_search(i)

        # Print and save network performance as desired
        if i%par['iters_per_output'] == 0:
            task_accuracy, full_accuracy = control.get_performance()
            loss_dict = control.get_losses_by_type(is_ranked)
            spikes    = control.get_spiking()

            task_loss = np.mean(loss_dict['task'][:par['num_survivors']])
            freq_loss = np.mean(loss_dict['freq'][:par['num_survivors']])
            reci_loss = np.mean(loss_dict['reci'][:par['num_survivors']])

            mean_loss = np.mean(loss[:par['num_survivors']])
            task_acc  = np.mean(task_accuracy[:par['num_survivors']])
            full_acc  = np.mean(full_accuracy[:par['num_survivors']])
            spiking   = np.mean(spikes[:par['num_survivors']])

            if par['learning_method'] in ['GA', 'TA']:
                top_task_acc = task_accuracy.max()
                top_full_acc = full_accuracy.max()
            elif par['learning_method'] == 'ES':
                top_task_acc = task_accuracy[0]
                top_full_acc = full_accuracy[0]

            save_record['iter'].append(i)
            save_record['top_task_acc'].append(top_task_acc)
            save_record['top_full_acc'].append(top_full_acc)
            save_record['mean_task_acc'].append(task_acc)
            save_record['mean_full_acc'].append(full_acc)
            save_record['loss'].append(mean_loss)
            save_record['loss_factors'].append(loss_dict)
            save_record['mut_str'].append(mutation_strength)
            save_record['spiking'].append(spiking)
            pickle.dump(save_record, open(par['save_dir']+par['save_fn']+'.pkl', 'wb'))
            if i%(10*par['iters_per_output']) == 0:
                print('Saving weights for iteration {}... ({})\n'.format(i, par['save_fn']))
                pickle.dump(to_cpu(control.var_dict), open(par['save_dir']+par['save_fn']+'_weights.pkl', 'wb'))

            status_stringA = 'Iter: {:4} | Task Loss: {:5.3f} | Freq Loss: {:5.3f} | Reci Loss: {:5.3f}'.format( \
                i, task_loss, freq_loss, reci_loss)
            status_stringB = 'Opt:  {:>4} | Full Loss: {:5.3f} | Mut Str: {:7.5f} | Spiking: {:5.2f} Hz'.format( \
                par['learning_method'], mean_loss, mutation_strength, spiking)
            status_stringC = 'S/O:  {:4} | Top Acc (Task/Full): {:5.3f} / {:5.3f}  | Mean Acc (Task/Full): {:5.3f} / {:5.3f}'.format( \
                int(time.time()-t0), top_task_acc, top_full_acc, task_acc, full_acc)
            print(status_stringA + '\n' + status_stringB + '\n' + status_stringC)
            t0 = time.time()

if __name__ == '__main__':
    main()
