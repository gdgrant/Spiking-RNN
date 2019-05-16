from imports import *


### Adaptive Leaky-Integrate-and-Fire spiking model

def run_lif(v, a, I, constants):
	""" Run one step of the Adaptive LIF algorithm """

	z, A, v_th = lif_spike(v, a, constants)
	v = lif_membrane(v, I, z, constants)
	a = lif_adaptation(a, z, constants)

	return v, a, z, A, v_th

def lif_membrane(v, I, z, c):
	""" Calculate the new membrane potential """

	return c['alpha']*v + I - z*c['v_th']

def lif_adaptation(a, z, c):
	""" Calculate the new threshold adaptation """

	return c['rho']*a + z * 0.25 # CHANGED

def lif_spike(v, a, c):
	""" Check potential thresholds for new spikes """

	A = c['v_th'] + c['beta']*a
	v_th = c['v_th']

	return heaviside((v-A)/v_th), A, v_th

def heaviside(x):
	""" Perform the Heaviside step function """

	return (x > 0).astype(x.dtype)
