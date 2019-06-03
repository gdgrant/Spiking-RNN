from imports import *


### Model utilities

def relu(x):
	""" Performs relu activation on x """
	return cp.maximum(0., x)


def softmax(x, a=-1):
	""" Performs stable softmax on x, across the last axis by default """
	c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
	return c/cp.sum(c, axis=a, keepdims=True)


def apply_EI(var, ei):
	""" Applies EI masking to a square variable, according to the given
		excitatory/inhibitory mask """
	return cp.matmul(ei, relu(var))


def cross_entropy(mask, target, output, eps=1e-16):
	""" Calculate the cross entropy loss for a rate-based network """
	return -cp.mean(mask[:,:,cp.newaxis]*target*cp.log(softmax(output)+eps))


def accuracy(output, target, mask, inc_fix=False):
	""" Calculate accuracy from output, target, and mask for the networks """

	arg_output = cp.argmax(output, -1)
	arg_target = cp.argmax(target, -1)
	mask = mask if inc_fix else mask * (arg_target != 0)

	return cp.sum(mask * (arg_output == arg_target))/cp.sum(mask)


def synaptic_plasticity(syn_x, syn_u, z, c, use_stp):
	""" Iterate the synaptic plasticity algorithm (if required) """

	if use_stp:
		syn_x = syn_x + c['alpha_std']*(1-syn_x) - syn_u*syn_x*z
		syn_u = syn_u + c['alpha_stf']*(c['U']-syn_u) + c['U']*(1-syn_u)*z
		syn_x = cp.clip(syn_x, 0., 1.)
		syn_u = cp.clip(syn_u, 0., 1.)

	return syn_x, syn_u


### Run environment utilities

def save_code():
	fns = [fn for fn in os.listdir('./') \
		if '.py' in fn and not '.pyc' in fn]

	file_dict = {}
	for fn in fns:
		with open('./'+fn, 'r') as f:
			file_dict[fn] = f.read()

	return file_dict
