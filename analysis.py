from imports import *
from utils import to_cpu
from sklearn.svm import SVC
from itertools import product

from parameters import par, update_parameters
from stimulus import Stimulus
from model import Model

savefn = 'saving_500neuron_120delay_v0_data_iter001900'
data = pickle.load(open('./savedir/archive/{}.pkl'.format(savefn), 'rb'))

update_parameters(data['par'], verbose=False)
update_parameters({'batch_size':128})
update_parameters({k+'_init':v for k,v in data['weights'].items()}, \
	verbose=False, update_deps=False)

trial_info = data['trial_info']
input_data = trial_info['neural_input']

end_dead_time       = par['dead_time']//par['dt']
end_fix_time        = end_dead_time + par['fix_time']//par['dt']
end_sample_time     = end_fix_time + par['sample_time']//par['dt']
end_delay_time      = end_sample_time + par['delay_time']//par['dt']

z = data['spiking']
v = data['voltage']
s = data['syn_x'] * data['syn_u']


################################################################################

def plot_activity():

	V_min = v[:,0,:].T.min()

	fig, ax = plt.subplots(3,1, figsize=(15,11), sharex=True)
	ax[0].imshow(z[:,0,:].T, aspect='auto')
	ax[0].set_title('Spiking')
	ax[1].imshow(v[:,0,:].T, aspect='auto', clim=(V_min,0.))
	ax[1].set_title('Membrane Voltage ($(V_r = {:5.3f}), {:5.3f} \\leq V_j^t \\leq 0$)'.format(par['adex']['V_r'].min(), V_min))
	ax[2].imshow(s[:,0,:].T, aspect='auto', clim=(0,1))
	ax[2].set_title('Synaptic Efficacy')

	ax[1].set_ylabel('Hidden Neuron')

	plt.show()


def match_to_pev_inputs(match):

	d_match = np.stack([match, 1-match, np.ones_like(match)], axis=-1)
	return d_match


def pev_analysis(a, b):

	b = b[:,np.newaxis]

	weights = np.linalg.lstsq(a, b, rcond=None)
	error   = b - a @ weights[0]

	error = error.reshape(b.shape)
	mse   = np.mean(error**2)
	rvar  = np.var(b)
	pev   = 1 - mse/(rvar+1e-9) if rvar > 1e-9 else 0

	return pev, weights[0]


def run_pev_analysis():

	### Run PEV analysis on the voltage and synaptic efficacy
	### to determine where the match/nonmatch information is stored

	d_match = match_to_pev_inputs(trial_info['match'])

	match_v_tuning = np.zeros([par['num_time_steps'], par['n_hidden']])
	match_s_tuning = np.zeros([par['num_time_steps'], par['n_hidden']])

	for n, t in product(range(par['n_hidden']), range(par['num_time_steps'])):
		print('N:{:>4}, T:{:>4}'.format(n,t), end='\r')

		pev_v, _ = pev_analysis(d_match, v[t,:,n])
		match_v_tuning[t,n] = pev_v

		pev_s, _ = pev_analysis(d_match, s[t,:,n])
		match_s_tuning[t,n] = pev_s


	fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True)
	ax[0].imshow(match_v_tuning[end_sample_time:end_delay_time].T, aspect='auto', clim=(0,1))
	ax[1].imshow(match_s_tuning[end_sample_time:end_delay_time].T, aspect='auto', clim=(0,1))

	ax[0].set_title('Voltage Match Tuning')
	ax[1].set_title('Syn. Eff. Match Tuning')

	ax[1].set_xlabel('Time (Since End of Sample, To Start of Test)')
	ax[1].set_ylabel('Neurons')

	plt.show()


def run_SVM_analysis():


	print('\nLoading and running model.')
	model = Model()
	stim  = Stimulus()
	runs  = 8

	m_all = []
	v_all = []
	s_all = []

	for i in range(runs):
		print('R:{:>2}'.format(i), end='\r')
		trial_info = stim.make_batch(var_delay=False)
		model.run_model(trial_info)

		m_all.append(trial_info['sample_cat'])
		v_all.append(to_cpu(model.v))
		s_all.append(to_cpu(model.s))

	del model
	del stim

	batch_size = runs*par['batch_size']

	m = np.concatenate(m_all, axis=0)
	v = np.concatenate(v_all, axis=1)
	s = np.concatenate(s_all, axis=1)

	print('Performing SVM decoding on {} trials.\n'.format(batch_size))
	# Initialize linear classifier
	args = {'kernel':'linear', 'decision_function_shape':'ovr', 'shrinking':False, 'tol':1e-3}
	lin_clf_v = SVC(**args)
	lin_clf_s = SVC(**args)

	score_v = np.zeros([par['num_time_steps']])
	score_s = np.zeros([par['num_time_steps']])

	# Choose training and testing indices
	train_pct = 0.75
	num_train_inds = int(batch_size * train_pct)

	shuffled   = np.random.permutation(batch_size)
	train_inds = shuffled[:num_train_inds]
	test_inds  = shuffled[num_train_inds:]

	for t in range(end_dead_time, par['num_time_steps']):
		print('T:{:>4}'.format(t), end='\r')
		
		lin_clf_v.fit(v[t,train_inds,:], m[train_inds])
		lin_clf_s.fit(s[t,train_inds,:], m[train_inds])

		dec_v = lin_clf_v.predict(v[t,test_inds,:])
		dec_s = lin_clf_s.predict(s[t,test_inds,:])

		score_v[t] = np.mean(m[test_inds]==dec_v)
		score_s[t] = np.mean(m[test_inds]==dec_s)


	fig, ax = plt.subplots(1, figsize=(12,8))
	ax.plot(score_v, c=[241/255, 153/255, 1/255], label='Voltage')
	ax.plot(score_s, c=[58/255, 79/255, 65/255], label='Syn. Eff.')

	ax.axhline(0.5, c='k', ls='--')
	ax.axvline(trial_info['timings'][0,0], c='k', ls='--')
	ax.axvline(trial_info['timings'][1,0], c='k', ls='--')

	ax.set_title('SVM Decoding of Sample Category')
	ax.set_xlabel('Time')
	ax.set_ylabel('Decoding Accuracy')
	ax.set_yticks([0., 0.25, 0.5, 0.75, 1.])
	ax.grid()
	ax.set_xlim(0,par['num_time_steps']-1)

	ax.legend()
	plt.savefig('./analysis/svm_decoding.png', bbox_inches='tight')

	print('SVM decoding complete.')

################################################################################

run_SVM_analysis()