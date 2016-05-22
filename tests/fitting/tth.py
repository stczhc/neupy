import numpy as np
from math import *
import random, sys, os, itertools
import re, json, dill

sys.path.insert(0, sys.path[0] + '/../..')
import neupy
from neupy import algorithms, layers
from neupy import environment

import theano
import theano.tensor as T
from sys import platform as _platform

if _platform == 'darwin':
  theano.config.cxx = "/usr/local/bin/g++-5"
  
coords = [[np.random.random() for i in range(0, 3)] for j in range(0, 8)]
coords = np.asarray(coords)

print coords

n = 5
xx = np.asarray(range(1, n + 1))
yy = np.zeros(n * (n - 1) / 2)
x = theano.shared(xx)
y = theano.shared(yy)
print y.get_value()

def lengths(cumu, x, y):
  y = T.set_subtensor(y[cumu[0]:cumu[0] + cumu[1] + 1], x[cumu[1]])
  return T.as_tensor_variable([cumu[0] + cumu[1] + 1, cumu[1] + 1])

g, gu = theano.scan(fn=lengths, outputs_info=T.zeros([2], dtype=int), 
  non_sequences=[x,y], n_steps=x.shape[0])

gf = theano.function([], g[-1], updates=gu)

print gf()
print y.get_value()

