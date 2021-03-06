from imports import *


def run_spike_model(V, w, I, spike_model, constants):

	if spike_model == 'adex':
		V, w, z = run_adex(V, w, I, constants)
	elif spike_model == 'izhi':
		V, w, z = run_izhi(V, w, I, constants)

	return V, w, cp.squeeze(z)


### Adaptive-Exponential spiking model
def run_adex(V, w, I, constants):
	""" Run one step of the AdEx algorithm """

	I_eff = I * constants['mu']

	V_next      = adex_membrane(V, w, I_eff, constants)
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




### Izhikevic Model spiking model

def run_izhi(V, w, I, c):
	""" Run one step of the Izhikevic spike model algorithm """

	I_eff = I * c['mu']

	V_next      = V + 0.04*V**2 + 5*V + 140 - w + I_eff
	w_next      = w + c['a']*(c['b'] * V - w)
	V, w, spike = izhi_spike(V_next, w_next, c)

	"""
	V, w, spike = izhi_spike(V, w, c)
	V      		= V + 0.04*V**2 + 5*V + 140 - w + I_eff
	w      		= w + c['a']*(c['b'] * V - w)
	"""
	return V, w, spike


def izhi_spike(V, w, c):
	""" Check potential thresholds for new spikes """

	spike = V > c['Vth']
	V = cp.where(spike, c['c'], V)
	w = cp.where(spike, w + c['d'], w)

	return V, w, spike
