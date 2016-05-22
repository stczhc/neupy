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
  
coords = [[[np.random.random() for i in range(0, 3)] for j in range(0, 8)] for j in range(0, 10)]
coords = np.asarray(coords)

print coords

def get_length_mat(n):
  m = n * (n - 1) / 2
  zz = np.zeros((m, n))
  mi = 0
  for i in range(0, n):
    for j in range(0, i):
      zz[mi, i] = 1
      zz[mi, j] = -1
      mi += 1
  return zz

x = T.dtensor3('x')
z = T.as_tensor_variable(get_length_mat(8))
lens = T.tensordot(x, z, axes=[1, 1]).norm(2, axis=1)
f = theano.function([x], lens)
print coords
y = f(coords)

def all_per(n):
  m = n * (n - 1) / 2
  zz = [ None ] * m
  zd = {}
  m = 0
  for i in range(0, n):
    for j in range(0, i):
      zz[m] = (i, j)
      zd[(i, j)] = m
      zd[(j, i)] = m
      m += 1
  lxr = [list(g) for g in itertools.permutations(range(0, n))]
  zzp = [ None ] * len(lxr)
  for i in range(len(lxr)):
    zr = [ None ] * len(zz)
    for j in range(len(zz)):
      zr[j] = zd[(lxr[i][zz[j][0]], lxr[i][zz[j][1]])]
    zzp[i] = zr
  return zzp

aa = all_per(8)

c = list(aa[0])
print c
print aa
for i in range(0, 100):
  random.shuffle(c)
  print c in aa
  
x = T.dmatrix('x')
y = x[:, [[2,0,1],[1,0,2]]]
f = theano.function([x], y)
print f(np.asarray([[9,4,7], [8,4,7]]))