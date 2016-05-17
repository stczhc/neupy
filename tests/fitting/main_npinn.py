
import numpy as np
from math import *
import random, sys, os, itertools, json, dill

sys.path.insert(0, sys.path[0] + '/../..')
import neupy
from neupy import algorithms, layers
from neupy import environment

import theano
import theano.tensor as T

print (neupy.__version__)
print (theano.config.base_compiledir)
print (os.getcwd())
environment.reproducible()
httoev = 27.21138505

# the homogeneous cluster
class Cluster(object):
  def __init__(self, n, expl):
    self.n = n
    self.atoms = np.zeros((n, 3))
    self.elems = [''] * n
    self.energy = 0.0
    self.mat_x = []
    self.mat_ll = []
    self.exp_length = expl
    ee = np.eye(n, dtype=int)
    for i in range(1, n + 1):
      lx = [list(g) for g in itertools.combinations(range(0, n), i)]
      lx = np.asarray(lx)
      self.mat_x.append(lx)
  
  # assign mat_ll values
  def gen_ll(self):
    ll = np.zeros((self.n, self.n))
    for i in range(0, self.n):
      for j in range(0, i):
        ll[i, j] = np.linalg.norm(self.atoms[i] - self.atoms[j])
    self.mat_ll = ll
  
  # generate lengths of all permutations of atoms
  def gen_per_ll(self, num):
    lx = [list(g) for g in itertools.permutations(range(0, num))]
    ll = []
    for g in lx:
      lc = []
      for i in range(0, num):
        for j in range(0, i):
          lc.append(exp(-np.linalg.norm(self.atoms[g[i]] - self.atoms[g[j]]) / self.exp_length))
      nr = len(lc)
      for i in range(0, nr):
        for j in range(0, i):
          lc.append(lc[i]*lc[j])
      ll.append(lc)
    return ll
  
  # flat of coordinates
  def get_flat(self):
    return self.atoms.reshape(self.n * 3)
  
  # only lengths, depend on mat_ll
  # if num is not None, then only part of atoms will be used to 
  # generate lengths. Those atoms are selected by all combinations
  def get_lengths_x(self, num=None):
    if num is None: num = self.n
    xx = self.mat_x[num - 1]
    ll = self.mat_ll
    r = []
    for k in range(0, xx.shape[0]):
      r.append([])
      for i in range(0, xx.shape[1]):
        for j in range(0, i):
          r[k].append(exp(-ll[xx[k, i], xx[k, j]] / self.exp_length))
    return np.asarray(r)
  
  # lengths and product of lengths
  def get_lengths_sp(self, num=None):
    r = self.get_lengths_x(num).tolist()
    for k in range(0, len(r)):
      ni = len(r[k])
      for i in range(0, ni):
        for j in range(0, i + 1): # added self-square here
          r[k].append(r[k][i] * r[k][j])
    return np.asarray(r)
  
  # return the position of the center
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  # will change the coordinates
  def center(self):
    self.atoms = self.atoms - self.get_center()
  
  # pre: an integer, only shuffle the first pre atoms
  # after shuffle, need to call gen_ll
  def shuffle(self, pre=None):
    if pre is None: pre = len(self.atoms)
    np.random.shuffle(self.atoms[0:pre])

# read input
def read_input(fn):
  json_data = open(fn).read()
  return json.loads(json_data)

# avoid overwritting
def new_file_name(x):
  i = 0
  y = x + '.' + str(i)
  while os.path.isfile(y):
    i += 1
    y = x + '.' + str(i)
  return y

def dump_data(name, obj):
  name = new_file_name(name)
  print 'dump data: ' + name
  with open(name, 'wb') as f:
    dill.dump(obj, f)

def load_data(name, i=None):
  if not i is None: name += '.' + str(i)
  print 'load data: ' + name
  with open(name, 'rb') as f:
    return dill.load(f)

# return a list of Cluster objects
def read_cluster(ener, xyz, ecut, expl, traj):
  max_energy = ecut
  f = open(ener, 'r')
  fs = f.readlines()
  f.close()
  lf = []
  for f in fs:
    g = f.replace('\n', '').split(' ')
    if len(g) >= 2:
      lf += [[g[0], float(g[1])]]
    else:
      lf += [[g[0]]]
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
      clu = Cluster(cn, expl=expl)
      i = 0
      clu.energy = float(fs[cc + 1][2]) if traj else l[1]
      for f in fs[cc + 2:cc + 2 + cn]:
        clu.elems[i] = f[0]
        ar = np.asarray([float(g) for g in f[1:4]])
        clu.atoms[i] = ar
        i = i + 1
      clu.center()
      if clu.energy < max_energy:
        clul.append(clu)
      cc += 2 + cn
  print 'structs loaded: ', len(clul)
  return clul

# transform training data, test data
# clus: the list of Cluster
# n: number of cases generated
# num: number of fitting degree
# typed: npic or fit
# npi_network: when fit, use this to transform input data
def trans_data(clus, n, num, typed, npi_network=None):
  x_d = []
  y_d = []
  for i in xrange(0, n):
    if i % (n / 100) == 0: print '{0} %'.format(i / (n / 100))
    idx = random.randint(len(clus))
    if typed == "fit":
      clu = clus[idx]
      clu.shuffle()
      clu.gen_ll()
      if npi_network is None:
        x_d.append(clu.get_lengths_x(num))
      else:
        x_d.append(npi_network.predict(clu.get_lengths_sp(num)))
      y_d.append(clu.energy)
    elif typed == "npic":
      idxg = random.randint(2)
      clu = clus[idx]
      clu.shuffle(num)
      pll = clu.gen_per_ll(num)
      if idxg == 0:
        idx = random.randint(len(pll))
        plx = [x for x in pll[idx]]
        random.shuffle(plx)
        while plx in pll:
          random.shuffle(plx)
        x_d.append([pll[0], plx])
        y_d.append([0, 1]) # different
      else:
        idx = random.randint(len(pll) - 1) + 1
        x_d.append([pll[0], pll[idx]])
        y_d.append([1, 0]) # the same
  x_d = np.asarray(x_d)
  y_d = np.asarray(y_d)
  return x_d, y_d

# only need to do this for energy
# since coordinates are transformed by exp
def find_max_min(clus, min_max_ext_ratio):
  ratio = min_max_ext_ratio
  y = np.asarray([x.energy for x in clus])
  dmax = np.ma.max(y)
  dmin = np.ma.min(y)
  dmm = dmax - dmin
  dmin -= dmm * ratio
  dmax += dmm * ratio
  return dmax, dmin

def trans_forward(ener, dmax, dmin):
  return (ener - dmin) / (dmax - dmin)

def trans_backward(ener, dmax, dmin):
  return ener * (dmax - dmin) + dmin

# when read global variables, no need to 
# use global keyword inside function defs.
layer_dic = {
  "tanh": layers.Tanh, 
  "sigmoid": layers.Sigmoid, 
  "soft_plus": layers.Softplus, 
  "soft_max": layers.Softmax, 
  "relu": layers.Relu, 
  "square_max": layers.SquareMax, 
  "abs_max": layers.AbsMax, 
  "diff_norm": layers.DiffNorm
}

# construct npi_comparing network
def create_network(ipdata, size, typed):
  x_layers = []
  ndim = 2 if typed == "npi" else 3
  # the structure of input:
  # npic: batch, sample 1/2, lengths
  # npi: batch, lengths -> piis
  # fit: batch, combinations, piis
  for i in range(len(ipdata["layer_types"])):
    if i == 0:
      l = layer_dic[ipdata["layer_types"][i]](size, ndim=ndim)
    else:
      size = ipdata["sizes"][i - 1]
      l = layer_dic[ipdata["layer_types"][i]](size)
    x_layers.append(l)
  if typed == "npic":
    x_layers += [ 
      layer_dic[ipdata["calibrate_method"]](presize=ipdata["sizes"][-1]), 
      layers.Reshape(), 
      layers.Root(1 / ipdata["exponential"]), 
      layers.Tanh(1), 
      layers.Softmax(2), 
      layers.ArgmaxOutput(2)
    ]
  elif typed == "npi":
    x_layers += [ layers.Output(size=ipdata["sizes"][-1]) ]
  elif typed == "fit":
    x_layers += [
      layers.Reshape(presize=ipdata["sizes"][-1]), 
      layers.Average(),
      layers.Output(1),
    ]
  opts = { "error": "mse", "step": ipdata["step"], "verbose": True, 
    "batch_size": ipdata["batch_size"], "nesterov": True, 
    "momentum": ipdata["momentum"], "shuffle_data": True, 
    "show_epoch": ipdata["show_epoch"] }
  return algorithms.Momentum(x_layers, **opts)

# nd is degree of fitting
def create_npic_network(ipdata, nd):
  size = nd * (nd - 1) / 2
  size = size + size * (size + 1) / 2
  return create_network(ipdata, size, "npic")

# nd is degree of fitting
def create_npi_network(ipdata, nd):
  size = nd * (nd - 1) / 2
  size = size + size * (size + 1) / 2
  return create_network(ipdata, size, "npi")

# ipsize is from the output size of npi network
def create_fit_network(ipdata, ipsize):
  return create_network(ipdata, ipsize, "fit")

# transfer weight and bias from neta to netb
def transfer_parameters(neta, netb):
  n = len(netb.layers)
  for i in range(n): 
    if isinstance(netb.layers[i], layers.ParameterBasedLayer):
      netb.layers[i].weight.set_value(neta.layers[i].weight.get_value())
      netb.layers[i].bias.set_value(neta.layers[i].bias.get_value())

npic_network_name = "npic_network.dill"
npic_data_name = "npic_data.dill"
npic_test_name = "npic_test.txt"
fit_network_name = "fit_network.dill"
fit_data_name = "fit_data.dill"
fit_test_name = "fit_test.txt"
fit_network_name = "fit_network.dill"

# main program
if __name__ == "__main__":
  if len(sys.argv) < 1 + 1:
    print ('Need input file!')
  else:
    ip = read_input(sys.argv[1])
    ipdt = ip["data_files"]
    ippn = ip["npi_network"]
    ipft = ip["fit_network"]
    
    print ('load primary data ...')
    rcopts = {
      "ener": ipdt["list_file"], "xyz": ipdt["struct_file"], 
      "ecut": ipdt["energy_cut"], "expl": ipdt["exp_length"], 
      "traj": ipdt["trajectory_form"]
    }
    clus = read_cluster(ener, xyz, ecut, expl, traj)
    dmax, dmin = find_max_min(clus, ipdt["min_max_ext_ratio"])
    random.shuffle(clus)
    nd = ipdt["degree_of_fitting"]
    
    for task in ip["task"]:
      # NPI comparing
      if task == "npic":
        print ('create network ...')
        if ippn["load_network"] != -1:
          if isinstance(ippn["load_network"], int):
            npic_net = load_data(name=npic_network_name, i=ippn["load_network"])
          else:
            npic_net = load_data(name=ippn["load_network"])
        else:
          npic_net = create_npic_network(ippn, nd)
        
        print ('transfrom data ...')
        if ipdt["load_data"] != -1:
          if isinstance(ipdt["load_data"], int):
            npic_data = load_data(name=npic_data_name, i=ipdt["load_data"])
          else:
            npic_data = load_data(name=ipdt["load_data"])
        else:
          ssr = sum(ipdt["sample_ratio"])
          nclus = len(clus)
          npic_data = []
          rstart = 0.0
          for i in range(0, 3):
            ratio = ipdt["sample_ratio"][i] / ssr
            rend = rstart + ratio
            nstart, nend = int(nclus * rstart), int(nclus * rend)
            rstart = rend
            x, y = trans_data(clus, ipdt["sample_number"][i], nd, typed="npic")
            npic_data += [x, y]
        if ipdt["dump_data"]:
          print ('dump data ...')
          dump_data(name=npic_data_name, obj=npic_data)
        
        print ('input data shape: ', npic_data[0].shape)
        print ('output data shape: ', npic_data[1].shape)
        print ('first input data: \n', npic_data[0])
        print ('first output data: \n', npic_data[1])
        
        if ippn["train_network"]:
          print ('train network ...')
          npic_net.train(*npic_data[0:4], epochs=ippn["epochs"])
        
        if ippn["test_network"]:
          print ('test network ...')
          ipd = npic_net.format_input_data(npic_data[4])
          nsqm = -1
          for i in range(len(npic_net.layers)):
            if isinstance(npic_net.layers[i], layers.SquareMax):
              nsqm = i
              break
          f = theano.function(inputs=[npic_net.variables.network_input],
            outputs=npic_net.layers[nsqm].prediction)
          npic_ind = f(ipd) # intermediate output
          npic_final = npic_net.predict(npic_data[4])
          npic_std = [x[1] for x in npic_data[5]]
          ntest = len(npic_final)
          nr = 0
          if ippn["test_output"]:
            ft = open(new_file_name(npic_test_name), 'w')
            ft.write('%8s %10s %10s %15s' % ('id', 'standard', 'predict', 'calibrate'))
          for idx, std, pre, ind in zip(range(ntest), npic_std, npic_final, npic_ind):
            if std == pre: nr += 1
            if ippn["test_output"]:
              ft.write('%8d %10d %10d %15.8f' % (idx, std, pre, ind))
          if ippn["test_output"]: ft.close()
          print (nr * 100 / ntest, '%')
        
        if ippn["dump_network"]:
          print ('dump network ...')
          dump_data(name=npic_network_name, obj=npic_net)
        
      # FIT energy
      elif task == "fit":
        print ('create network ...')
        if ipft["load_network"] != -1:
          if isinstance(ipft["load_network"], int):
            fit_net = load_data(name=fit_network_name, i=ipft["load_network"])
          else:
            fit_net = load_data(name=ipft["load_network"])
        else:
          fit_net = create_fit_network(ipft, ippn["sizes"][-1])
        
        print ('transfrom data ...')
        if ipdt["load_data"] != -1:
          if isinstance(ipdt["load_data"], int):
            fit_data = load_data(name=fit_data_name, i=ipdt["load_data"])
          else:
            fit_data = load_data(name=ipdt["load_data"])
        else:
          print ('create npi network ...')
          if ipft["load_npic_network"] != -1:
            if isinstance(ipft["load_npic_network"], int):
              npic_net = load_data(name=npic_network_name, i=ipft["load_npic_network"])
            else:
              npic_net = load_data(name=ippn["load_network"])
            npi_net = create_npi_network(ippn, nd)
            transfer_parameters(npic_net, npi_net)
          else:
            npi_net = None
          
          ssr = sum(ipdt["sample_ratio"])
          nclus = len(clus)
          fit_data = []
          rstart = 0.0
          for i in range(0, 3):
            ratio = ipdt["sample_ratio"][i] / ssr
            rend = rstart + ratio
            nstart, nend = int(nclus * rstart), int(nclus * rend)
            rstart = rend
            x, y = trans_data(clus, ipdt["sample_number"][i], nd, 
              typed="fit", npi_network=npi_net)
            y = trans_forward(y, dmax, dmin)
            fit_data += [x, y]
        if ipdt["dump_data"]:
          print ('dump data ...')
          dump_data(name=fit_data_name, obj=fit_data)
        
        print ('input data shape: ', fit_data[0].shape)
        print ('output data shape: ', fit_data[1].shape)
        print ('first input data: \n', fit_data[0])
        print ('first output data: \n', fit_data[1])
        
        if ipft["train_network"]:
          print ('train network ...')
          fit_net.train(*fit_data[0:4], epochs=ipft["epochs"])
        
        if ipft["test_network"]:
          print ('test network ...')
          fit_pre = fit_net.predict(fit_data[4])
          fit_pre = [f[0] for f in fit_pre]
          fit_std = fit_data[5]
          fit_std = trans_backward(fit_std, dmax, dmin)
          fit_pre = trans_backward(fit_pre, dmax, dmin)
          ntest = len(fit_pre)
          nr = 0.0
          if ipft["test_output"]:
            ft = open(new_file_name(fit_test_name), 'w')
            ft.write('%8s %15s %15s %15s' % ('id', 'standard', 'predict', 'error'))
          for idx, std, pre in zip(range(ntest), fit_std, fit_pre):
            nr += (std - pre) ** 2
            if ipft["test_output"]:
              ft.write('%8d %15.8f %15.8f %15.8f' % (idx, std, pre, abs(std - pre)))
          if ipft["test_output"]: ft.close()
          print ('%15.8f' % sqrt(nr / ntest) * httoev, ' eV')
        
        if ipft["dump_network"]:
          print ('dump network ...')
          dump_data(name=fit_network_name, obj=fit_net)
