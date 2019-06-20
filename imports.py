# Basic tools
import os, sys, time
import pickle
import itertools

# Plotting
if len(sys.argv) > 1:
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Numerical processing
import numpy as np
if len(sys.argv) > 1:
	import cupy as cp
	import cupy.linalg as LA
	cp.cuda.Device(sys.argv[1]).use()
else:
	import numpy.linalg as LA
	cp = np
