from imports import *
from gpu_utils import to_gpu, to_cpu
from parameters import par

def output_behavior(it, match_info, timings, y):

	match = np.where(match_info)[0]
	nonmatch = np.where(np.logical_not(match_info))[0]
	time = np.arange(par['num_time_steps'])

	y_match        = to_cpu(cp.mean(y[:,match,:], axis=1))
	y_nonmatch     = to_cpu(cp.mean(y[:,nonmatch,:], axis=1))

	y_match_err    = to_cpu(cp.std(y[:,match,:], axis=1))
	y_nonmatch_err = to_cpu(cp.std(y[:,nonmatch,:], axis=1))

	c_res = [[60/255, 21/255, 59/255, 1.0], [164/255, 14/255, 76/255, 1.0], [77/255, 126/255, 168/255, 1.0]]
	c_err = [[60/255, 21/255, 59/255, 0.5], [164/255, 14/255, 76/255, 0.5], [77/255, 126/255, 168/255, 0.5]]

	fig, ax = plt.subplots(2,1, figsize=[16,8], sharex=True)
	for i, (r, e) in enumerate(zip([y_match, y_nonmatch], [y_match_err, y_nonmatch_err])):
		err_low  = r - e
		err_high = r + e

		ax[i].fill_between(time, err_low[:,0], err_high[:,0], color=c_err[0])
		ax[i].fill_between(time, err_low[:,1], err_high[:,1], color=c_err[1])
		ax[i].fill_between(time, err_low[:,2], err_high[:,2], color=c_err[2])

		ax[i].plot(time, r[:,0], c=c_res[0], label='Fixation')
		ax[i].plot(time, r[:,1], c=c_res[1], label='Cat. 1 / Match')
		ax[i].plot(time, r[:,2], c=c_res[2], label='Cat. 2 / Non-Match')

		ax[i].legend(loc="upper left")

		ax[i].axvline(timings[0,:].min(), c='k', ls='--')
		ax[i].axvline(timings[1,:].max(), c='k', ls='--')

	fig.suptitle('Output Neuron Behavior')
	ax[0].set_title('Cat. 1 / Match Trials')
	ax[1].set_title('Cat. 2 / Non-Match Trials')

	ax[0].set_ylabel('Mean Response')
	ax[1].set_ylabel('Mean Response')
	ax[1].set_xlim(time.min(), time.max())
	ax[1].set_xlabel('Time')

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
	ax[3].set_title('Membrane Voltage ($(V_r = {:5.3f}), {:5.3f} \\leq V_j^t \\leq 0$)'.format(par['adex']['V_r'].min(), V_min))

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