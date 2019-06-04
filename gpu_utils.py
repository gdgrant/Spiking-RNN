from imports import *


### GPU utilities

def to_gpu(x):
	""" Move numpy arrays (or dicts/lists of arrays) to GPU """
	if len(sys.argv) > 1:
		if type(x) == dict:
			return {k:to_gpu(a) for (k, a) in x.items()}
		elif type(x) == list:
			return [to_gpu(a) for a in x]
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