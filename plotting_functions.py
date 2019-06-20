from imports import *
from utils import to_gpu, to_cpu
from parameters import par

def clopath_update_plot(it, cl_in, cl_rnn, gr_in, gr_rnn):

	update_list = to_cpu([cl_in, cl_rnn, gr_in, gr_rnn])
	update_name = ['Clopath W_in', 'Clopath W_rnn', 'Grad W_in', 'Grad W_rnn']

	fig, ax = plt.subplots(2,2, figsize=[12,10])
	for i, j in itertools.product([0,1], [0,1]):
		im = ax[i,j].imshow(update_list[i+2*j], aspect='auto')
		ax[i,j].set_title(update_name[i+2*j])
		fig.colorbar(im, ax=ax[i,j])

	plt.savefig('./savedir/{}_clopath{:0>6}.png'.format(par['savefn'], it), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_clopath{:0>6}.pdf'.format(par['savefn'], it), bbox_inches='tight')
	plt.clf()
	plt.close()


def plot_grads_and_epsilons(it, trial_info, model, h, eps_v_rec, eps_w_rec, eps_ir_rec):

	h = to_cpu(h[:,0,:])
	eps_v_rec = to_cpu(eps_v_rec)
	eps_w_rec = to_cpu(eps_w_rec)
	eps_ir_rec = to_cpu(eps_ir_rec)

	V_min = to_cpu(model.v[:,0,:,:].T.min())

	fig, ax = plt.subplots(8, 1, figsize=[16,22], sharex=True)

	ax[0].imshow(trial_info['neural_input'][:,0,:].T, aspect='auto')
	ax[0].set_title('Input Data')
	ax[0].set_ylabel('Input Neuron')

	ax[1].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
	ax[1].set_title('Spiking')
	ax[1].set_ylabel('Hidden Neuron')

	ax[2].plot(to_cpu(model.z[:,0,0]), label='Spike')
	ax[2].plot(to_cpu(model.v[:,0,0,0]) * -10, label='- Voltage x 10')
	ax[2].plot(h[:,0], label='Gradient')
	ax[2].legend()
	ax[2].set_title('Single Neuron')

	ax[3].imshow(h.T, aspect='auto', clim=(0, par['gamma_psd']))
	ax[3].set_title('Pseudogradient (${} \\leq h \\leq {}$) | Sum: $h = {:6.3f}$'.format(0., par['gamma_psd'], np.sum(h)))
	ax[3].set_ylabel('Hidden Neuron')

	ax[4].imshow(to_cpu(model.v[:,0,0,:].T), aspect='auto')
	ax[4].set_title('Membrane Voltage ($(V_r = {:5.3f}), {:5.3f} \\leq V_j^t \\leq 0$)'.format(par[par['spike_model']]['V_r'].min(), V_min))
	ax[4].set_ylabel('Hidden Neuron')

	ax[5].imshow(eps_v_rec.T, aspect='auto')
	ax[5].set_title('Voltage Eligibility (${:6.3f} \\leq e_{{v,rec}} \\leq {:6.3f}$)'.format(eps_v_rec.min(), eps_v_rec.max()))
	ax[5].set_ylabel('Hidden Neuron')

	ax[6].imshow(eps_w_rec.T, aspect='auto')
	ax[6].set_title('Adaptation Eligibility (${:6.3f} \\leq e_{{w,rec}} \\leq {:6.3f}$)'.format(eps_w_rec.min(), eps_w_rec.max()))
	ax[6].set_ylabel('Hidden Neuron')

	ax[7].imshow(eps_ir_rec.T, aspect='auto')
	ax[7].set_title('Current Eligibility (${:6.3f} \\leq e_{{ir,rec}} \\leq {:6.3f}$)'.format(eps_ir_rec.min(), eps_ir_rec.max()))
	ax[7].set_ylabel('Hidden Neuron')

	# ax[0,1].imshow(trial_info['neural_input'][:,0,:].T, aspect='auto')
	# ax[1,1].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
	# ax[2,1].imshow(h.T, aspect='auto', clim=(0, par['gamma_psd']))
	# ax[3,1].imshow(to_cpu(model.v[:,0,0,:].T), aspect='auto')

	# ax[4,1].imshow(eps_v_rec.T, aspect='auto')
	# ax[4,1].set_xlabel('Time')

	# for i in range(4):
	# 	ax[i,0].set_xticks([])

	# for i in range(5):
	# 	ax[i,1].set_xlim(200,350)

	plt.savefig('./savedir/{}_epsilon_iter{:0>6}.png'.format(par['savefn'], it), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_epsilon_iter{:0>6}.pdf'.format(par['savefn'], it), bbox_inches='tight')
	plt.clf()
	plt.close()


def output_behavior(it, trial_info, y):


	if par['task'] == 'dmswitch':
		task_info = trial_info['task']
		task_names = ['dms', 'dmc']
		num_tasks = 2
		height = 14
	else:
		task_names = [par['task']]
		num_tasks = 1
		height = 8

	match_info, timings = trial_info['match'], trial_info['timings']

	fig, ax = plt.subplots(2*num_tasks, 1, figsize=[16,height], sharex=True)

	for task in range(num_tasks):

		if par['task'] == 'dmswitch':
			task_mask = (task_info == task)
			match = np.where(np.logical_and(task_mask, match_info))[0]
			nonmatch = np.where(np.logical_and(task_mask, np.logical_not(match_info)))[0]

		else:
			match = np.where(match_info)[0]
			nonmatch = np.where(np.logical_not(match_info))[0]

		time = np.arange(par['num_time_steps'])

		y_match        = to_cpu(cp.mean(y[:,match,:], axis=1))
		y_nonmatch     = to_cpu(cp.mean(y[:,nonmatch,:], axis=1))

		y_match_err    = to_cpu(cp.std(y[:,match,:], axis=1))
		y_nonmatch_err = to_cpu(cp.std(y[:,nonmatch,:], axis=1))

		c_res = [[60/255, 21/255, 59/255, 1.0], [164/255, 14/255, 76/255, 1.0], [77/255, 126/255, 168/255, 1.0]]
		c_err = [[60/255, 21/255, 59/255, 0.5], [164/255, 14/255, 76/255, 0.5], [77/255, 126/255, 168/255, 0.5]]

		for i, (r, e) in enumerate(zip([y_match, y_nonmatch], [y_match_err, y_nonmatch_err])):
			j = 2*task + i

			err_low  = r - e
			err_high = r + e

			ax[j].fill_between(time, err_low[:,0], err_high[:,0], color=c_err[0])
			ax[j].fill_between(time, err_low[:,1], err_high[:,1], color=c_err[1])
			ax[j].fill_between(time, err_low[:,2], err_high[:,2], color=c_err[2])

			ax[j].plot(time, r[:,0], c=c_res[0], label='Fixation')
			ax[j].plot(time, r[:,1], c=c_res[1], label='Cat. 1 / Match')
			ax[j].plot(time, r[:,2], c=c_res[2], label='Cat. 2 / Non-Match')

			for t in range(timings.shape[0]):
				ax[j].axvline(timings[t,:].min(), c='k', ls='--')

	fig.suptitle('Output Neuron Behavior')
	for task in range(num_tasks):
		j = task*2
		ax[j].set_title('Task: {} | Cat. 1 / Match Trials'.format(task_names[task].upper()))
		ax[j+1].set_title('Task: {} | Cat. 2 / Non-Match Trials'.format(task_names[task].upper()))

	for j in range(2*num_tasks):
		ax[j].legend(loc="upper left")
		ax[j].set_ylabel('Mean Response')
	ax[0].set_xlim(time.min(), time.max())
	ax[2*num_tasks-1].set_xlabel('Time')

	plt.savefig('./savedir/{}_outputs_iter{:0>6}.png'.format(par['savefn'], it), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_outputs_iter{:0>6}.pdf'.format(par['savefn'], it), bbox_inches='tight')
	plt.clf()
	plt.close()


def visualize_delta(i, var_dict, grad_dict):

	for n in [k for k in grad_dict.keys() if 'rnn' in k]:
		fig, ax = plt.subplots(1,2, figsize=[16,8])
		im = ax[0].imshow(to_cpu(par['learning_rate']*grad_dict[n]), aspect='auto')
		fig.colorbar(im, ax=ax[0])
		im = ax[1].imshow(to_cpu(var_dict[n]), aspect='auto')
		fig.colorbar(im, ax=ax[1])

		fig.suptitle(n)
		ax[0].set_title('Gradient')
		ax[1].set_title('Variable')

		plt.savefig('./savedir/{}_delta_{}_iter{:0>6}.png'.format(par['savefn'], n, i), bbox_inches='tight')
		if par['save_pdfs']:
			plt.savefig('./savedir/{}_delta_{}_iter{:0>6}.pdf'.format(par['savefn'], n, i), bbox_inches='tight')
		plt.clf()
		plt.close()


def activity_plots(i, model):

	V_min = to_cpu(model.v[:,0,:,:].T.min())

	fig, ax = plt.subplots(4,1, figsize=(15,11), sharex=True)
	ax[0].imshow(to_cpu(model.input_data[:,0,:].T), aspect='auto')
	ax[0].set_title('Input Data')
	ax[1].imshow(to_cpu((model.input_data[:,0,:] @ model.eff_var['W_in']).T), aspect='auto')
	ax[1].set_title('Projected Inputs')
	ax[2].imshow(to_cpu(model.z[:,0,:].T), aspect='auto')
	ax[2].set_title('Spiking')
	ax[3].imshow(to_cpu(model.v[:,0,0,:].T), aspect='auto', clim=(V_min,0.))
	ax[3].set_title('Membrane Voltage ($(V_r = {:5.3f}), {:5.3f} \\leq V_j^t \\leq 0$)'.format(par[par['spike_model']]['V_r'].min(), V_min))

	ax[0].set_ylabel('Input Neuron')
	ax[1].set_ylabel('Hidden Neuron')
	ax[2].set_ylabel('Hidden Neuron')
	ax[3].set_ylabel('Hidden Neuron')

	plt.savefig('./savedir/{}_activity_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_activity_iter{:0>6}.pdf'.format(par['savefn'], i), bbox_inches='tight')
	plt.clf()
	plt.close()


def training_curve(i, iter_record, full_acc_record, task_acc_record):
	
	fig, ax = plt.subplots(1,1, figsize=(8,8))
	ax.plot(iter_record, full_acc_record, label='Full Accuracy')
	ax.plot(iter_record, task_acc_record, label='Match/Nonmatch Accuracy')
	ax.axhline(0.5, c='k', ls='--', label='Match/Nonmatch Chance Level')
	ax.legend(loc='upper left')
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Accuracy')
	ax.set_title('Accuracy Training Curve')
	ax.set_ylim(0,1)
	ax.set_xlim(0,i)
	ax.grid()

	plt.savefig('./savedir/{}_training_curve_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_training_curve_iter{:0>6}.pdf'.format(par['savefn'], i), bbox_inches='tight')
	plt.clf()
	plt.close()


def run_pev_analysis(sample, syn_eff, z, I_sqr_record, i):

	### Run PEV analysis on the voltage and synaptic efficacy
	### to determine where the match/nonmatch information is stored

	def pev_analysis(a, b):
		weights = np.linalg.lstsq(a, b, rcond=None)
		error   = b - a @ weights[0]

		error = error.reshape(b.shape)
		mse   = np.mean(error**2)
		rvar  = np.var(b)
		pev   = 1 - mse/(rvar+1e-9) if rvar > 1e-9 else 0

		return pev, weights[0]

	syn_eff = np.squeeze(syn_eff)

	filtered_z = np.zeros_like(z)
	alpha = 0.98
	for t in range(1, z.shape[0]):
		filtered_z[t, :, :] = alpha*filtered_z[t-1, :, :] + (1-alpha)*z[t,:,:]

	sample_dir = np.ones((par['batch_size'], 3))
	sample_dir[:,1] = np.cos(2*np.pi*sample/par['num_motion_dirs'])
	sample_dir[:,2] = np.sin(2*np.pi*sample/par['num_motion_dirs'])

	pev_z = np.zeros([par['num_time_steps'], par['n_hidden']])
	pev_syn = np.zeros([par['num_time_steps'], par['n_hidden']])

	for n, t in product(range(par['n_hidden']), range(par['num_time_steps'])):

		pev_z[t,n], _ = pev_analysis(sample_dir, filtered_z[t,:,n,np.newaxis])
		pev_syn[t,n], _ = pev_analysis(sample_dir, syn_eff[t,:,n,np.newaxis])


	fig, ax = plt.subplots(2,2, figsize=(10,8))
	ax[0,0].imshow(pev_z.T, aspect='auto', clim=(0,1))
	ax[0,1].imshow(pev_syn.T, aspect='auto', clim=(0,1))
	ax[1,0].plot(np.percentile(pev_z, 95, axis=1),'g', label = 'spikes 95pct')
	ax[1,0].plot(np.percentile(pev_syn, 95, axis=1),'m', label = 'synapses 95pct')
	ax[1,0].plot(np.percentile(pev_z, 80, axis=1),'g--', label = 'spikes 80pct')
	ax[1,0].plot(np.percentile(pev_syn, 80, axis=1),'m--', label = 'synapses 80pct')
	ax[1,0].legend()
	ax[1,1].plot(I_sqr_record)

	ax[0,0].set_title('Spike PEV')
	ax[0,1].set_title('Synaptic PEV')

	ax[1,0].set_xlabel('Time (ms)')
	ax[1,1].set_xlabel('Iterations')
	ax[1,0].set_ylabel('PEV')
	ax[0,0].set_ylabel('Neurons')


	plt.savefig('./savedir/{}_pev{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_pev{:0>6}.pdf'.format(par['savefn'], i), bbox_inches='tight')
	plt.clf()
	plt.close()


def EI_testing_plots(i, I_sqr_record, W_rnn_grad_sum_record, W_rnn_grad_norm_record):

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



def run_pev_analysis(sample, syn_eff, z, I_sqr_record, i):

	def pev_analysis(a, b):

		weights = np.linalg.lstsq(a, b, rcond=None)
		error   = b - a @ weights[0]

		error = error.reshape(b.shape)
		mse   = np.mean(error**2)
		rvar  = np.var(b)
		pev   = 1 - mse/(rvar+1e-9) if rvar > 1e-9 else 0

		return pev, weights[0]

	### Run PEV analysis on the voltage and synaptic efficacy
	### to determine where the match/nonmatch information is stored

	syn_eff = np.squeeze(syn_eff)

	filtered_z = np.zeros_like(z)
	alpha = 0.98
	for t in range(1, z.shape[0]):
		filtered_z[t, :, :] = alpha*filtered_z[t-1, :, :] + (1-alpha)*z[t,:,:]

	sample_dir = np.ones((par['batch_size'], 3))
	sample_dir[:,1] = np.cos(2*np.pi*sample/par['num_motion_dirs'])
	sample_dir[:,2] = np.sin(2*np.pi*sample/par['num_motion_dirs'])

	pev_z = np.zeros([par['num_time_steps'], par['n_hidden']])
	pev_syn = np.zeros([par['num_time_steps'], par['n_hidden']])

	for n, t in itertools.product(range(par['n_hidden']), range(par['num_time_steps'])):

		pev_z[t,n], _ = pev_analysis(sample_dir, filtered_z[t,:,n,np.newaxis])
		pev_syn[t,n], _ = pev_analysis(sample_dir, syn_eff[t,:,n,np.newaxis])


	fig, ax = plt.subplots(2,2, figsize=(10,8))
	ax[0,0].imshow(pev_z.T, aspect='auto', clim=(0,1))
	ax[0,1].imshow(pev_syn.T, aspect='auto', clim=(0,1))
	ax[1,0].plot(np.percentile(pev_z, 95, axis=1),'g', label = 'spikes 95pct')
	ax[1,0].plot(np.percentile(pev_syn, 95, axis=1),'m', label = 'synapses 95pct')
	ax[1,0].plot(np.percentile(pev_z, 80, axis=1),'g--', label = 'spikes 80pct')
	ax[1,0].plot(np.percentile(pev_syn, 80, axis=1),'m--', label = 'synapses 80pct')
	ax[1,0].legend()
	ax[1,1].plot(I_sqr_record)

	ax[0,0].set_title('Spike PEV')
	ax[0,1].set_title('Synaptic PEV')

	ax[1,0].set_xlabel('Time (ms)')
	ax[1,1].set_xlabel('Iterations')
	ax[1,0].set_ylabel('PEV')
	ax[0,0].set_ylabel('Neurons')


	plt.savefig('./savedir/{}_pev_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
	if par['save_pdfs']:
		plt.savefig('./savedir/{}_pev_iter{:0>6}.pdf'.format(par['savefn'], i), bbox_inches='tight')
	plt.clf()
	plt.close()