from imports import *

### Helper functions

def L2_norm(x):
	return cp.sqrt(cp.sum(cp.square(x)))


def clip_by_norm(x, n):
	L = L2_norm(x)
	if L > n:
		return x * n / L
	else:
		return x


### Optimizers

class Standard:

	def __init__(self, var_dict, learning_rate):

		self.var_dict = var_dict
		self.lr = learning_rate


	def apply_gradients(self, grad_dict):

		# Apply gradient updates
		for n in self.var_dict.keys():
			self.var_dict[n] += self.lr * grad_dict[n]

		return self.var_dict


class AdamOpt:

	def __init__(self, var_dict, learning_rate=0.001, beta1=0.9, \
			beta2=0.999, epsilon=1e-8):

		self.lr       = learning_rate
		self.beta1    = beta1
		self.beta2    = beta2
		self.t        = 0
		self.epsilon  = epsilon
		self.var_dict = var_dict

		self.m = {}
		self.v = {}
		self.delta_grads = {}
		for name, var in self.var_dict.items():
			self.m[name] = cp.zeros_like(var)
			self.v[name] = cp.zeros_like(var)
			self.delta_grads[name] = cp.zeros_like(var)


	def reset_params(self):

		self.t = 0
		for name, var in self.var_dict.items():
			self.m[name] = cp.zeros_like(var)
			self.v[name] = cp.zeros_like(var)
			self.delta_grads[name] = cp.zeros_like(var)


	def apply_gradients(self, grad_dict):

		self.t += 1
		lr = self.lr * cp.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)

		for n in self.var_dict.keys():
			new_m = self.beta1*self.m[n] + (1-self.beta1)*grad_dict[n]
			new_v = self.beta2*self.v[n] + (1-self.beta2)*grad_dict[n]**2

			delta_grad = (lr*new_m)/(cp.sqrt(new_v) + self.epsilon)
			delta_grad = clip_by_norm(delta_grad, 1.)

			self.m[n] = new_m
			self.v[n] = new_v

			self.var_dict[n] += delta_grad

		return self.var_dict

