from imports import *


### GPU utilities

def to_gpu(x):
	""" Move numpy arrays (or dicts/lists of arrays) to GPU """
	if len(sys.argv) > 1:
		if type(x) == dict:
			return {k:to_gpu(a) for (k, a) in x.items()}
		elif type(x) == list:
			return [to_gpu(a) for a in x]
		elif type(x) == tuple:
			return to_gpu(list(x))
		else:
			if type(x) == np.ndarray and x.dtype == np.float64:
				return cp.asarray(x.astype(np.float32))
			else:
				return cp.asarray(x)
	else:
		return x

def to_cpu(x):
	""" Move cupy arrays (or dicts/lists of arrays) to CPU """
	if len(sys.argv) > 1:
		if type(x) == dict:
			return {k:to_cpu(a) for (k, a) in x.items()}
		elif type(x) == list:
			return [to_cpu(a) for a in x]
		else:
			return cp.asnumpy(x)
	else:
		return x


### Precision initializations

def cp_zeros(shape):
	return cp.zeros(shape, dtype=cp.float32)


def np_zeros(shape):
	return np.zeros(shape, dtype=np.float32)


def cp_ones(shape):
	return cp.ones(shape, dtype=cp.float32)


def np_ones(shape):
	return np.ones(shape, dtype=np.float32)