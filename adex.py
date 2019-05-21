from imports import *


### Adaptive-Exponential spiking model


def run_adex(V, w, I, constants):
	""" Run one step of the AdEx algorithm """

	V_next      = adex_membrane(V, w, I, constants)
	w_next      = adex_adaptation(V, w, constants)
	V, w, spike = adex_spike(V_next, w_next, constants)

	return V, w, spike


def adex_membrane(V, w, I, c):
	""" Calculate the new membrane potential """


	term1 = I + c['g']*c['D']*cp.exp((V-c['V_T'])/c['D'])
	term2 = w + c['g']*(V-c['E'])
	return V + (c['dt']/c['C'])*(term1-term2)


def adex_adaptation(V, w, c):
	""" Calculate the new adaptation current """

	term1 = c['a']*(V-c['E'])
	term2 = w
	return w + (c['dt']/c['tau'])*(term1-term2)


def adex_spike(V, w, c):
	""" Check potential thresholds for new spikes """

	spike = V > c['Vth']
	V = cp.where(spike, c['V_r'], V)
	w = cp.where(spike, w + c['b'], w)

	return V, w, spike
