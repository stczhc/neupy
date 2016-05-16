
import numpy as np
from math import *
import random

import sys
import os
sys.path.insert(0, sys.path[0] + '/../..')
import neupy
from neupy import algorithms, layers
from neupy import environment
print neupy.__version__

import theano
print theano.config.base_compiledir
# theano.config.exception_verbosity = 'high'
import theano.tensor as T
import itertools

tdr = os.environ['TMPDIR']
print os.getcwd()
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
  
  def gen_per_ll(self, num):
    lx = [list(g) for g in itertools.permutations(range(0, num))]
    ll = []
    for g in lx:
      lc = []
      for i in range(0, num):
        for j in range(0, i):
          lc.append(exp(-np.linalg.norm(self.atoms[g[i]] - self.atoms[g[j]])/5.0))
      nr = len(lc)
      for i in range(0, nr):
        for j in range(0, i):
          lc.append(lc[i]*lc[j])
      ll.append(lc)
    return ll
  
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  def get_flat(self):
    return self.atoms.reshape(self.n * 3)
  
  # only lengths
  def get_lengths_x(self, num=None):
    if num is None: num = self.n
    xx = self.mat_x[num - 1]
    ll = self.mat_ll
    r = []
    for k in range(0, xx.shape[0]):
      r.append([])
      for i in range(0, xx.shape[1]):
        for j in range(0, i):
          r[k].append(exp(-ll[xx[k, i], xx[k, j]]/5.0))
    return np.asarray(r)
  
  # lengths and product of lengths
  def get_lengths_sp(self, num=None):
    r = self.get_lengths_x(num).tolist()
    for k in range(0, len(r)):
      ni = len(r[k])
      for i in range(0, ni):
        for j in range(0, i):
          r[k].append(r[k][i] * r[k][j])
    return np.asarray(r)
  
  def get_lengths(self):
    return self.get_lengths_x()
  
  def center(self):
    self.atoms = self.atoms - self.get_center()
  
  def shuffle(self, pre=None):
    if pre is None: pre = len(self.atoms)
    np.random.shuffle(self.atoms[0:pre])
  
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
  max_energy = -954.6
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

def load_data(clus, n, num=None, network=None):
  x_d = []
  y_d = []
  for i in xrange(0, n):
    if i % (n / 100) == 0: print '{0} %'.format(i / (n / 100))
    idx = np.random.randint(len(clus))
    clu = clus[idx]
    clu.shuffle()
    clu.gen_ll()
    if network is None:
      x_d.append(clu.get_lengths_sp(num))
    else:
      x_d.append(network.predict(clu.get_lengths_sp(num)))
    y_d.append(clu.energy)
  y_d = np.array(y_d)
  x_d = np.asarray(x_d)
  return x_d, y_d

def load_data_cmp(clus, n, num=None):
  x_d = []
  y_d = []
  for i in xrange(0, n):
    if i % (n / 100) == 0: print '{0} %'.format(i / (n / 100))
    mm = 0.01
    idxg = np.random.randint(2)
    if idxg == 0:
      idx = np.random.randint(len(clus))
      clu = clus[idx]
      clu.shuffle(num)
      pll = clu.gen_per_ll(num)
      idx = np.random.randint(len(pll))
      plx = [x for x in pll[idx]]
      random.shuffle(plx)
      while plx in pll:
        random.shuffle(plx)
      x_d.append([pll[0], plx])
      y_d.append([1,0]) # different
    else:
      idx = np.random.randint(len(clus))
      clu = clus[idx]
      clu.shuffle(num)
      pll = clu.gen_per_ll(num)
      idx = np.random.randint(len(pll) - 1) + 1
      if idxg == 1:
        x_d.append([pll[0], pll[idx]])
      y_d.append([0,1]) # the same
  y_d = np.array(y_d)
  x_d = np.asarray(x_d)
  return x_d, y_d

def find_max_min(clus):
  ratio = 0.05
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
    # c.atoms = (c.atoms - dmin[0]) / (dmax[0] - dmin[0])
    c.energy = (c.energy - dmin[1]) / (dmax[1] - dmin[1])

def trans_backward(clus, dmax, dmin):
  for c in clus:
    # c.atoms = c.atoms * (dmax[0] - dmin[0]) + dmin[0]
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

def store_data(dat):
  c = new_file_name('data/train_data.dill')
  print 'data dump at ' + c
  with open(c, 'wb') as f:
    dill.dump(dat, f)

def load_data(i):
  if isinstance(i, str):
    c = i
  else:
    c = 'data/train_data.dill.' + str(i)
  print 'data load at ' + c
  with open(c, 'rb') as f:
    return dill.load(f)

def load(i):
  if isinstance(i, str):
    c = i
  else:
    c = 'data/network-storage.dill.' + str(i)
  print 'load at ' + c
  with open(c, 'rb') as f:
    return dill.load(f)

class ACT(layers.ActivationLayer):
    # activation_function = (lambda x:T.nnet.relu(x) * 2 - 1)
    # activation_function = (lambda x:T.nnet.sigmoid(x) * 2 - 1)
    # activation_function = (lambda x:T.tanh(x/2) + T.nnet.sigmoid(x))
    # activation_function = (lambda x:T.nnet.sigmoid(x))
    activation_function = T.tanh
    # activation_function = T.nnet.relu
    # activation_function = (lambda x: (x/5)**2)
    # activation_function = (lambda x: T.nnet.relu(x) + 2*T.nnet.sigmoid(x*5))

print 'load data ...'
lcmp = False
lstore_data = True
lload_data = False
print 'lcmp = ', lcmp
clus = read_cluster('./data/tm_pt8/list.txt', './data/tm_pt8/structs/final_#.xyz', traj=True)
dmax, dmin = find_max_min(clus)
trans_forward(clus, dmax, dmin)
random.shuffle(clus)
for c in clus: c.gen_ll()
ratio = 9.0 / 10.0
if lcmp:
  if lload_data:
    x_train, y_train, x_test, y_test = load_data(-1)
  else:
    x_train, y_train = load_data_cmp(clus[1:int(len(clus)*ratio)], 100000, num=5)
    x_test, y_test = load_data_cmp(clus[int(len(clus)*ratio):], 10000, num=5)
  if lstore_data: store_data((x_train, y_train, x_test, y_test))
else:
  if lload_data:
    x_train, y_train, x_test, y_test = load_data(-1)
  else:
    old_network = load('data/nip.dill.2')
    pre_network = network = algorithms.Momentum(
      [
        ACT(old_network.layers[0].size, ndim=2), # 28 x 1 -> 28 x 50
        ACT(100), 
        layers.Output(20)
      ])
    for i in range(0, 1):
      pre_network.layers[i].weight.set_value(old_network.layers[i].weight.get_value())
      pre_network.layers[i].bias.set_value(old_network.layers[i].bias.get_value())
    x_train, y_train = load_data(clus[1:int(len(clus)*ratio)], 100000, num=5, network=pre_network)
    x_test, y_test = load_data(clus[int(len(clus)*ratio):], 10000, num=5, network=pre_network)
  if lstore_data: store_data((x_train, y_train, x_test, y_test))

print len(x_train), len(y_train), len(x_test), len(y_test)

environment.reproducible()

load_i = -1
load_sim = False
print 'construct network ...'
if not lcmp:
  network = algorithms.Momentum(
    [
      ACT(x_train.shape[-1], ndim=3), # 28 x 1 -> 28 x 50
      ACT(100), # 28 x 50 -> 28 x 1
      ACT(50), # 28 x 50 -> 28 x 1
      layers.Softplus(10), 
      layers.Reshape(presize=4), # 28 x 1 -> 28
      layers.Average(), # 28 -> 1
      layers.Output(1), 
    ],
    # error='binary_crossentropy',
    error='mse',
    step=0.1,
    verbose=True,
    batch_size = 100,
    # mu=0.1,
    # mu_update_factor = 1,
    # addons=[algorithms.WeightDecay], 
    nesterov = True,
    momentum = 0.8, 
    shuffle_data=True,
    # decay_rate = 0.0001, 
    show_epoch = 2
  ) if load_i == -1 else load(load_i)
else:
  network = algorithms.Momentum(
    [
      ACT(x_train.shape[-1], ndim=3), # 28 x 1 -> 28 x 50
      ACT(100), 
      layers.SquareMax(presize=50), 
      layers.Reshape(), 
      layers.Root(1/10.0), 
      layers.Tanh(1), 
      layers.Softmax(2), 
      layers.ArgmaxOutput(2), 
    ],
    # error='binary_crossentropy',
    error='mse',
    step=0.005,
    verbose=True,
    batch_size = 10,
    # mu=0.1,
    # mu_update_factor = 1,
    # addons=[algorithms.WeightDecay], 
    nesterov = True,
    momentum = 0.8, 
    shuffle_data=True,
    # decay_rate = 0.0001, 
    show_epoch = 2
  ) if load_i == -1 else load(load_i)

print network
print 'train ...'
if lcmp:
  print zip(x_train[0:5], y_train[0:5])
else:
  print (x_train[0], y_train[0])
if not load_sim:
  network.train(x_train, y_train, x_test, y_test, epochs=500)

y_pre = network.predict(x_test)
if lcmp:
  input_data = network.format_input_data(x_test)
  f = theano.function(inputs=[network.variables.network_input],
    outputs=network.layers[2].prediction)
  y_pre2 = f(input_data)
else:
  y_pre2 = y_pre

if not lcmp:
  y_pre = trans_backward_y(y_pre, dmax, dmin)
  y_test = trans_backward_y(y_test, dmax, dmin)

import math
httoev = 27.21138505
res = 0.0
ff = open('data/fitenergy.txt', 'w')
ntotal = len(y_pre)
nr = 0
for x, y, y2 in zip(y_test, y_pre, y_pre2):
  # if lcmp: y = [y,1] if y > 0.5 else [y,0]
  print x, y, y2
  if lcmp:
    if not load_sim:
      if x[y] == 1: nr += 1
  if not lcmp: ff.write('%15.6f %15.6f\n' % (x, y[0]))
  # if lcmp: ff.write('%15.6f %15.6f\n' % (x[0], y[0]))
  if not lcmp: res += (x - y[0])**2
ff.close()
if lcmp:
  print nr * 100 / ntotal, '%'
else:
  print math.sqrt(res/len(y_test)) * httoev

store(network)
