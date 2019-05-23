from imports import *
from gpu_utils import to_cpu
from sklearn.svm import SVC
from itertools import product

from parameters import par, update_parameters
from stimulus import Stimulus
from model import Model

# List save files to analyze
savefns = [
	'verify_500neuron_var120delay_v0_data_iter000300',
]

# Make plot
fig, ax = plt.subplots(1, 2, figsize=(14,6))

# Iterate over provided save files
for num_fn, fn in enumerate(savefns):

	print('Processing file {} of {}.'.format(num_fn+1, len(savefns)))

	# Load data
	data = pickle.load(open('./savedir/{}.pkl'.format(fn), 'rb'))

	# Update parameters with current weights
	update_parameters(data['par'], verbose=False)
	update_parameters({'batch_size':128}, verbose=False)
	update_parameters({k+'_init':v for k,v in data['weights'].items()}, \
		verbose=False, update_deps=False)

	end_dead_time       = par['dead_time']//par['dt']
	end_fix_time        = end_dead_time + par['fix_time']//par['dt']
	end_sample_time     = end_fix_time + par['sample_time']//par['dt']
	end_delay_time      = end_sample_time + par['delay_time']//par['dt']

	# Make a new model and stimulus (which use the loaded parameters)
	print('\nLoading and running model.')
	model = Model()
	stim  = Stimulus()
	runs  = 2

	c_all = []
	d_all = []
	v_all = []
	s_all = []

	# Run a couple batches to generate sufficient data points
	for i in range(runs):
		print('R:{:>2}'.format(i), end='\r')
		trial_info = stim.make_batch(var_delay=False)
		model.run_model(trial_info, testing=True)

		c_all.append(trial_info['sample_cat'])
		d_all.append(trial_info['sample_dir'])
		v_all.append(to_cpu(model.v))
		s_all.append(to_cpu(model.s))

	del model
	del stim

	batch_size = runs*par['batch_size']

	c = np.concatenate(c_all, axis=0)
	d = np.concatenate(d_all, axis=0)
	v = np.concatenate(v_all, axis=1)
	s = np.concatenate(s_all, axis=1)

	print('Model run complete.')
	print('Performing SVM decoding on {} trials.\n'.format(batch_size))

	# Initialize linear classifiers
	args = {'kernel':'linear', 'decision_function_shape':'ovr', \
		'shrinking':False, 'tol':1e-3}
	lin_clf_cv = SVC(**args)
	lin_clf_cs = SVC(**args)
	lin_clf_dv = SVC(**args)
	lin_clf_ds = SVC(**args)

	c_score_v = np.zeros([par['num_time_steps']])
	c_score_s = np.zeros([par['num_time_steps']])
	d_score_v = np.zeros([par['num_time_steps']])
	d_score_s = np.zeros([par['num_time_steps']])

	# Choose training and testing indices
	train_pct = 0.75
	num_train_inds = int(batch_size * train_pct)

	shuffled   = np.random.permutation(batch_size)
	train_inds = shuffled[:num_train_inds]
	test_inds  = shuffled[num_train_inds:]

	# Fit the classifiers for each time step,
	# and judge their accuracy at predicting
	# category encoding.
	for t in range(end_dead_time, par['num_time_steps']):
		print('T:{:>4}'.format(t), end='\r')
		
		lin_clf_cv.fit(v[t,train_inds,:], c[train_inds])
		lin_clf_cs.fit(s[t,train_inds,:], c[train_inds])
		lin_clf_dv.fit(v[t,train_inds,:], d[train_inds])
		lin_clf_ds.fit(s[t,train_inds,:], d[train_inds])

		dec_cv = lin_clf_cv.predict(v[t,test_inds,:])
		dec_cs = lin_clf_cs.predict(s[t,test_inds,:])
		dec_dv = lin_clf_dv.predict(v[t,test_inds,:])
		dec_ds = lin_clf_ds.predict(s[t,test_inds,:])

		c_score_v[t] = np.mean(c[test_inds]==dec_cv)
		c_score_s[t] = np.mean(c[test_inds]==dec_cs)
		d_score_v[t] = np.mean(d[test_inds]==dec_dv)
		d_score_s[t] = np.mean(d[test_inds]==dec_ds)

	# Plot classification scores
	if num_fn == 0:
		ax[0].plot(c_score_v, c=[241/255, 153/255, 1/255], label='Voltage')
		ax[0].plot(c_score_s, c=[58/255, 100/255, 65/255], label='Syn. Eff.')
	else:
		ax[0].plot(c_score_v, c=[241/255, 153/255, 1/255])
		ax[0].plot(c_score_s, c=[58/255, 100/255, 65/255])

	ax[1].plot(d_score_v, c=[241/255, 153/255, 1/255])
	ax[1].plot(d_score_s, c=[58/255, 100/255, 65/255])

	print('Processing of file {} complete.'.format(num_fn+1))

# Decorate and save plots
ax[0].axhline(0.5, c='k', ls='--')
ax[1].axhline(0.125, c='k', ls='--')
ax[0].axvline(trial_info['timings'][0,0], c='k', ls='--')
ax[0].axvline(trial_info['timings'][1,0], c='k', ls='--')
ax[1].axvline(trial_info['timings'][0,0], c='k', ls='--')
ax[1].axvline(trial_info['timings'][1,0], c='k', ls='--')

ax[0].set_title('Sample Category')
ax[1].set_title('Sample Direction')
for i in range(2):
	ax[i].set_xlabel('Time')
	ax[i].set_ylabel('Decoding Accuracy')
	ax[i].set_xlim(0,par['num_time_steps']-1)
	ax[i].set_yticks([0., 0.25, 0.5, 0.75, 1.])
	ax[i].grid()

ax[0].legend(loc='lower right')
fig.suptitle('SVM Task Decoding from Neural Population')

plt.savefig('./analysis/svm_decoding.png', bbox_inches='tight')
plt.savefig('./analysis/svm_decoding.pdf', bbox_inches='tight')
