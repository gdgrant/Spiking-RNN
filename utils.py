import sys, time, pickle
import itertools

import numpy as np
if len(sys.argv) > 1:
    import cupy as cp
    cp.cuda.Device(sys.argv[1]).use()
else:
    cp = np


### GPU utilities

def to_gpu(x):
    """ Move numpy arrays (or dicts of arrays) to GPU """
    if type(x) == dict:
        return {k:cp.asarray(a) for (k, a) in x.items()}
    else:
        return cp.asarray(x)

def to_cpu(x):
    """ Move cupy arrays (or dicts of arrays) to CPU """
    if len(sys.argv) > 1:
        if type(x) == dict:
            return {k:cp.asnumpy(a.astype(cp.float32)) for (k, a) in x.items()}
        else:
            return cp.asnumpy(x.astype(cp.float32))
    else:
        if type(x) == dict:
            return {k:a.astype(cp.float32) for (k, a) in x.items()}
        else:
            return x.astype(cp.float32)


### Network functions

def relu(x):
    """ Performs relu on x """
    return cp.maximum(0., x)

def softmax(x, a=-1):
    """ Performs stable softmax on x, across the last axis by default """
    c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
    return c/cp.sum(c, axis=a, keepdims=True).astype(cp.float32)

def apply_EI(var, ei):
    """ Applies EI masking to a square variable, according to the given
        excitatory/inhibitory mask """
    return cp.matmul(relu(var), ei)

def synaptic_plasticity(h, syn_x, syn_u, constants, use_stp, hidden_size):
    """ If required, applies STP updates to the hidden state and STP
        variables.  If not required, just ensures correct hidden shape. """

    if use_stp:
        syn_x += constants['alpha_std']*(1-syn_x) - constants['stp_mod']*syn_u*syn_x*h
        syn_u += constants['alpha_stf']*(constants['U']-syn_u) + constants['stp_mod']*constants['U']*(1-syn_u)*h
        syn_x = cp.minimum(1., relu(syn_x))
        syn_u = cp.minimum(1., relu(syn_u))
        h_post = syn_u*syn_x*h
    else:
        h_post = h*cp.ones([1,1,hidden_size], dtype=cp.float32)

    return h_post, syn_x, syn_u
    

### Judgement functions

def cross_entropy(mask, target, output, eps=1e-16):
    """ Calculate the cross entropy loss for a rate-based network """
    mask   = mask.astype(cp.float32)
    target = target.astype(cp.float32)
    output = output.astype(cp.float32)

    return -cp.mean(mask[:,:,cp.newaxis]*target*cp.log(softmax(output)+eps)).astype(cp.float32)


### Reporting functions

def accuracy(output, target, mask, inc_fix=False):
    """ Calculate accuracy from output, target, and mask for the networks """
    output = output.astype(cp.float32)
    target = target.astype(cp.float32)
    mask   = mask.astype(cp.float32)

    arg_output = cp.argmax(output, -1)
    arg_target = cp.argmax(target, -1)
    mask = mask if inc_fix else mask * (arg_target != 0)

    acc = cp.sum(mask * (arg_output == arg_target))/cp.sum(mask)

    return acc.astype(cp.float32)
