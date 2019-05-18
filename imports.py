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
	cp.cuda.Device(sys.argv[1]).use()
else:
	cp = np
