
import numpy as np
from math import *

import sys
sys.path.insert(0, sys.path[0] + '/../..')
import neupy
print neupy.__version__

class Cluster(object):
  def __init__(self, n):
    self.n = n
    self.atoms = np.zeros((n, 3))
    self.elems = [''] * n
    self.energy = 0.0
  
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  def get_flat(self):
    return self.atoms.reshape(self.n * 3)
  
  def get_lengths(self):
    ll = []
    for i in range(0, self.n):
      for j in range(0, i):
        ll.append(np.linalg.norm(self.atoms[i]-self.atoms[j]))
    return np.asarray(ll)
  
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

def read_cluster(ener, xyz):
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
    clu = Cluster(int(fs[0][0]))
    i = 0
    clu.energy = l[1]
    for f in fs[2:]:
      clu.elems[i] = f[0]
      ar = np.asarray([float(g) for g in f[1:]])
      clu.atoms[i] = ar
      i = i + 1
    clu.center()
    clul.append(clu)
  return clul

def load_data(n):
  clus = read_cluster('./data/energyx.txt', './data/ck_structs/pos_#.xyz')
  x_d = []
  y_d = []
  for i in xrange(0, n):
    idx = np.random.randint(len(clus))
    clu = clus[idx]
    clu.shuffle()
    clu.rotate()
    x_d.append(clu.atoms)
    # x_d.append(clu.get_flat())
    y_d.append(clu.energy)
  y_d = np.array(y_d)
  x_d = np.asarray(x_d)
  return x_d, y_d

def find_max_min(x_train, y_train):
  ratio = 0.1
  dmax = [np.ma.max(x_train), np.ma.max(y_train)]
  dmin = [np.ma.min(x_train), np.ma.min(y_train)]
  for i in range(0, 2):
    dmm = dmax[i] - dmin[i]
    dmin[i] -= dmm * ratio
    dmax[i] += dmm * ratio
  return dmax, dmin

def trans_forward(x, y, dmax, dmin):
  x = (x - dmin[0]) / (dmax[0] - dmin[0])
  y = (y - dmin[1]) / (dmax[1] - dmin[1])
  # x = x * 2 - 1
  # y = y * 2 - 1
  return x, y

def trans_backward(x, y, dmax, dmin):
  # x = (x + 1) / 2
  # y = (y + 1) / 2
  x = x * (dmax[0] - dmin[0]) + dmin[0]
  y = y * (dmax[1] - dmin[1]) + dmin[1]
  return x, y

print 'load data ...'
x_train, y_train = load_data(60000)
x_test, y_test = load_data(10000)

print 'transform data ...'
dmax, dmin = find_max_min(x_train, y_train)
x_train, y_train = trans_forward(x_train, y_train, dmax, dmin)
x_test, y_test = trans_forward(x_test, y_test, dmax, dmin)

from neupy import algorithms, layers, __version__
import theano.tensor as T

from neupy import environment
environment.reproducible()

print __version__
import os
print os.getcwd()
#exit(0)

import theano

theano.config.exception_verbosity = 'high'

class ACT(layers.ActivationLayer):
    # activation_function = (lambda x:T.nnet.relu(x) * 2 - 1)
    # activation_function = (lambda x:T.nnet.sigmoid(x) * 2 - 1)
    # activation_function = (lambda x:T.tanh(x/2) + T.nnet.sigmoid(x))
    # activation_function = (lambda x:T.nnet.sigmoid(x))
    # activation_function = T.tanh
    activation_function = T.nnet.relu
    # activation_function = (lambda x: (x/5)**2)
    # activation_function = (lambda x: T.nnet.relu(x) + 2*T.nnet.sigmoid(x*5))

print 'construct network ...'
# network = algorithms.LevenbergMarquardt(
# network = algorithms.MinibatchGradientDescent(
network = algorithms.Momentum(
  [
    # layers.Combination(num = 8, comb = 2), 
    # layers.Reshape((28, 6)), # 28 x 2 x 3 -> 28 x 6
    # ACT(6), # 28 x 6 -> 28 x 4
    # # ACT(50), # 28 x 10 -> 28 x 4
    # ACT(28, dotdim = 0, presize = 4), # 28 x 4 -> 1 x 4
    # layers.Reshape(presize=4), # 1 x 4 -> 4
    # ACT(16), # 4 -> 4
    # layers.Softplus(4),
    # layers.Output(1),
    
    # ACT(8, ndim=3, dotdim=0), # 8 x 3 -> 4 x 3
    # # layers.Square(3, presize=50), # 50 x 3 -> 50 x 1
    # ACT(3, presize=30), # 50 x 3 -> 50 x 1
    # layers.Reshape(presize=1), # 50 x 1 -> 50
    # ACT(30), 
    # layers.Softplus(10),
    # layers.Output(1),
    
    # ACT(24), 
    # ACT(50), 
    # layers.Square(15), 
    # layers.Softplus(10), 
    # layers.Output(1), 
    
    # layers.Combination(num = 8, comb = 2), # 8 x 3 -> 28 x 2 x 3
    # layers.Length2D(), # 28 x 2 x 3 -> 28
    # layers.Reshape((28, 1)), # 28 -> 28 x 1
    # ACT(1), # 28 x 1 -> 28 x 50
    # ACT(50), # 28 x 50 -> 28 x 1
    # # layers.Softplus(4), 
    # layers.Reshape(presize=1), # 28 x 1 -> 28
    # layers.Average(), # 28 -> 1
    # # ACT(28), 
    # layers.Output(1), 
    
    # layers.Combination(num = 8, comb = 3), # 8 x 3 -> 56 x 3 x 3
    # layers.Length3D(), # 56 x 3 x 3 -> 56 x 3
    # # layers.Reshape((28, 1)), # 28 -> 28 x 1
    # ACT(3), # 28 x 1 -> 28 x 50
    # ACT(50), # 28 x 50 -> 28 x 1
    # ACT(20), # 28 x 50 -> 28 x 1
    # layers.Softplus(4), 
    # layers.Reshape(presize=1), # 28 x 1 -> 28
    # layers.Average(), # 28 -> 1
    # # ACT(28), 
    # layers.Output(1), 
    
    layers.Combination(num = 8, comb = 4), # 8 x 3 -> 56 x 3 x 3
    layers.Length4D(), # 56 x 3 x 3 -> 56 x 3
    # layers.Reshape((28, 1)), # 28 -> 28 x 1
    ACT(6), # 28 x 1 -> 28 x 50
    ACT(100), # 28 x 50 -> 28 x 1
    ACT(30), # 28 x 50 -> 28 x 1
    layers.Softplus(4), 
    layers.Reshape(presize=1), # 28 x 1 -> 28
    layers.Average(), # 28 -> 1
    # ACT(28), 
    layers.Output(1), 
  ],
  # error='binary_crossentropy',
  error='mse',
  step=0.2,
  verbose=True,
  batch_size = 20,
  # mu=0.1,
  # mu_update_factor = 1,
  nesterov = True,
  momentum = 0.7, 
  shuffle_data=True,
  show_epoch = 2
)

print 'train ...'
network.train(x_train, y_train, x_test, y_test, epochs=200)

y_pre = network.predict(x_test)

_, y_pre = trans_backward(x_test, y_pre, dmax, dmin)
x_test, y_test = trans_backward(x_test, y_test, dmax, dmin)

import math
httoev = 27.21138505
res = 0.0
for x, y in zip(y_test, y_pre):
  res += (x - y)**2
print math.sqrt(res/len(y_test)) * httoev
