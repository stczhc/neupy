
import numpy as np
from math import *
import random

import sys
import os
sys.path.insert(0, sys.path[0] + '/../..')
import neupy
print neupy.__version__

import theano
print theano.config.base_compiledir
# theano.config.exception_verbosity = 'high'
import theano.tensor as T
import itertools

tdr = os.environ['TMPDIR']
# os.environ['THEANO_FLAGS'] = 'base_compiledir=' + tdr

class Cluster(object):
  def __init__(self, n):
    self.n = n
    self.atoms = np.zeros((n, 3))
    self.elems = [''] * n
    self.energy = 0.0
    self.mat_f = []
    self.mat_x = []
    self.mat_ll = []
    ee = np.eye(n, dtype=int)
    for i in range(1, n + 1):
      le = [list(g) for g in itertools.combinations(ee, i)]
      lx = [list(g) for g in itertools.combinations(range(0, n), i)]
      le = np.asarray(le)
      lx = np.asarray(lx)
      self.mat_f.append(lambda x,le=le: np.dot(le, x))
      self.mat_x.append(lx)

  def gen_ll(self):
    ll = np.zeros((self.n, self.n))
    for i in range(0, self.n):
      for j in range(0, i):
        ll[i, j] = np.linalg.norm(self.atoms[i] - self.atoms[j])
    self.mat_ll = ll
  
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  def get_flat(self):
    return self.atoms.reshape(self.n * 3)
  
  def get_lengths_x(self, num=None):
    if num is None: num = self.n
    xx = self.mat_x[num - 1]
    ll = self.mat_ll
    r = []
    for k in range(0, xx.shape[0]):
      r.append([])
      for i in range(0, xx.shape[1]):
        for j in range(0, i):
          r[k].append(ll[xx[k, i], xx[k, j]])
    return np.asarray(r)
  
  def get_lengths_sp(self, num=None):
    if num is None: num = self.n
    ll = self.mat_ll
    xx = self.mat_x[num - 1]
    r = []
    for l in range(0, xx.shape[0]):
      r.append([])
      for i in range(0, xx.shape[1]):
        for j in range(0, i):
          lij = ll[xx[l, i], xx[l, j]]
          for k in range(0, j):
            lik = ll[xx[l, i], xx[l, k]]
            ljk = ll[xx[l, j], xx[l, k]]
            r[l] += [lij + lik, lij + ljk, lik + ljk, 
              lij * lik, lij * ljk, lik * ljk]
    return np.asarray(r)
  
  def get_lengths(self):
    return self.get_lengths_x()
  
  def center(self):
    self.atoms = self.atoms - self.get_center()
  
  def shuffle(self):
    np.random.shuffle(self.atoms)
  
  def rotate(self):
    theta = np.random.random() * 2.0 * pi
    the = np.random.random() * pi
    phi = np.random.random() * 2.0 * pi
    u = [sin(the) * cos(phi), sin(the) * sin(phi), cos(the)]
    r = np.array([[cos(theta) + u[0]**2 * (1 - cos(theta)), 
      u[0]*u[1] * (1 - cos(theta)) - u[2] * sin(theta), 
      u[0]*u[2] * (1 - cos(theta)) + u[1] * sin(theta)], 
      [u[1]*u[0] * (1 - cos(theta)) + u[2] * sin(theta), 
      cos(theta) + u[1]**2 * (1 - cos(theta)), 
      u[1]*u[2] * (1 - cos(theta)) - u[0] * sin(theta)], 
      [u[2]*u[0] * (1 - cos(theta)) - u[1] * sin(theta), 
      u[2]*u[1] * (1 - cos(theta)) + u[0] * sin(theta), 
      cos(theta) + u[2]**2 * (1 - cos(theta))]])
    self.atoms = self.atoms.dot(r.T)

def read_cluster(ener, xyz, traj=False):
  max_energy = -954.46
  f = open(ener, 'r')
  fs = f.readlines()
  f.close()
  lf = []
  for f in fs:
    g = f.replace('\n', '').split(' ')
    lf += [[g[0], float(g[1])]]
  clul = []
  for l in lf:
    fn = xyz.replace('#', l[0])
    f = open(fn, 'r')
    fs = f.readlines()
    f.close()
    fs = [[g for g in f.replace('\n', '').split(' ') if len(g) != 0] for f in fs]
    cc = 0
    while cc < len(fs):
      cn = int(fs[cc][0])
      clu = Cluster(cn)
      i = 0
      clu.energy = float(fs[cc+1][2]) if traj else l[1]
      for f in fs[cc+2:cc+2+cn]:
        clu.elems[i] = f[0]
        ar = np.asarray([float(g) for g in f[1:4]])
        clu.atoms[i] = ar
        i = i + 1
      clu.center()
      if clu.energy < max_energy:
        clul.append(clu)
      cc += 2 + cn
  print 'struct loaded: ', len(clul)
  return clul

def load_data(clus, n, num=None):
  x_d = []
  y_d = []
  for i in xrange(0, n):
    if i % (n / 100) == 0: print '{0} %'.format(i / (n / 100))
    idx = np.random.randint(len(clus))
    clu = clus[idx]
    clu.shuffle()
    x_d.append(clu.get_lengths_x(num))
    y_d.append(clu.energy)
  y_d = np.array(y_d)
  x_d = np.asarray(x_d)
  return x_d, y_d

def find_max_min(clus):
  ratio = 0.1
  x_train = np.asarray([x.atoms for x in clus])
  y_train = np.asarray([x.energy for x in clus])
  dmax = [np.ma.max(x_train), np.ma.max(y_train)]
  dmin = [np.ma.min(x_train), np.ma.min(y_train)]
  for i in range(0, 2):
    dmm = dmax[i] - dmin[i]
    dmin[i] -= dmm * ratio
    dmax[i] += dmm * ratio
  return dmax, dmin

def trans_forward(clus, dmax, dmin):
  for c in clus:
    c.atoms = (c.atoms - dmin[0]) / (dmax[0] - dmin[0])
    c.energy = (c.energy - dmin[1]) / (dmax[1] - dmin[1])

def trans_backward(clus, dmax, dmin):
  for c in clus:
    c.atoms = c.atoms * (dmax[0] - dmin[0]) + dmin[0]
    c.energy = c.energy * (dmax[1] - dmin[1]) + dmin[1]

def trans_backward_y(y, dmax, dmin):
  return y * (dmax[1] - dmin[1]) + dmin[1]

def new_file_name(x):
  i = 0
  y = x + '.' + str(i)
  while os.path.isfile(y):
    i += 1
    y = x + '.' + str(i)
  return y

import dill
def store(network):
  c = new_file_name('data/network-storage.dill')
  print 'dump at ' + c
  with open(c, 'wb') as f:
    dill.dump(network, f)

def load(i):
  print 'load at ' + 'data/network-storage.dill.' + str(i)
  with open('data/network-storage.dill.' + str(i), 'rb') as f:
    return dill.load(f)

print 'load data ...'
clus = read_cluster('./data/energyx.txt', './data/ck_structs/pos_#.xyz')
dmax, dmin = find_max_min(clus)
trans_forward(clus, dmax, dmin)
random.shuffle(clus)
for c in clus: c.gen_ll()
ratio = 9.0 / 10.0
x_train, y_train = load_data(clus[1:int(len(clus)*ratio)], 10000, num=5)
x_test, y_test = load_data(clus[int(len(clus)*ratio):], 2000, num=5)

from neupy import algorithms, layers, __version__

from neupy import environment
environment.reproducible()

print __version__
print os.getcwd()

class ACT(layers.ActivationLayer):
    # activation_function = (lambda x:T.nnet.relu(x) * 2 - 1)
    # activation_function = (lambda x:T.nnet.sigmoid(x) * 2 - 1)
    # activation_function = (lambda x:T.tanh(x/2) + T.nnet.sigmoid(x))
    # activation_function = (lambda x:T.nnet.sigmoid(x))
    # activation_function = T.tanh
    activation_function = T.nnet.relu
    # activation_function = (lambda x: (x/5)**2)
    # activation_function = (lambda x: T.nnet.relu(x) + 2*T.nnet.sigmoid(x*5))

load_i = -1
print 'construct network ...'
network = algorithms.Momentum(
  [
    ACT(x_train.shape[-1], ndim=3), # 28 x 1 -> 28 x 50
    ACT(100), # 28 x 50 -> 28 x 1
    ACT(50), # 28 x 50 -> 28 x 1
    layers.Softplus(6), 
    layers.Reshape(presize=2), # 28 x 1 -> 28
    layers.Average(), # 28 -> 1
    layers.Output(1), 
  ],
  # error='binary_crossentropy',
  error='mse',
  step=0.15,
  verbose=True,
  batch_size = 20,
  # mu=0.1,
  # mu_update_factor = 1,
  nesterov = True,
  momentum = 0.8, 
  shuffle_data=True,
  show_epoch = 10
) if load_i == -1 else load(load_i)

print network
print 'train ...'
print x_train[0], y_train[0]
network.train(x_train, y_train, x_test, y_test, epochs=1000)

y_pre = network.predict(x_test)

y_pre = trans_backward_y(y_pre, dmax, dmin)
y_test = trans_backward_y(y_test, dmax, dmin)

import math
httoev = 27.21138505
res = 0.0
ff = open('data/fitenergy.txt', 'w')
for x, y in zip(y_test, y_pre):
  # print x, y
  ff.write('%15.6f %15.6f\n' % (x, y[0]))
  res += (x - y)**2
ff.close()
print math.sqrt(res/len(y_test)) * httoev

store(network)
