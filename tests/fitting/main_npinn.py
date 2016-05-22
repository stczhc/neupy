
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

print (neupy.__version__)
print (theano.config.base_compiledir)
print (os.getcwd())
environment.reproducible()
httoev = 27.21138505

un = 8
mat_x = []
mat_xr = []
ee = np.eye(un, dtype=int)
for i in range(1, un + 1):
  lx = [list(g) for g in itertools.combinations(range(0, un), i)]
  lx = np.asarray(lx)
  mat_x.append(lx)
  lxr = [list(g) for g in itertools.permutations(range(0, i))]
  lxr = np.asarray(lxr)
  mat_xr.append(lxr)

# the homogeneous cluster
class Cluster(object):
  def __init__(self, n, expl):
    self.n = n
    self.atoms = np.zeros((n, 3))
    self.elems = [''] * n
    self.energy = 0.0
    self.mat_ll = []
    self.exp_length = expl
  
  # assign mat_ll values
  def gen_ll(self):
    ll = np.zeros((self.n, self.n))
    for i in range(0, self.n):
      for j in range(0, i):
        ll[i, j] = np.linalg.norm(self.atoms[i] - self.atoms[j])
    self.mat_ll = ll
  
  # generate lengths of all permutations of atoms
  def gen_per_ll(self, num):
    lx = mat_xr[num - 1]
    ll = []
    for g in lx:
      lc = []
      if self.exp_length != 0:
        for i in range(0, num):
          for j in range(0, i):
            lc.append(exp(-np.linalg.norm(self.atoms[g[i]] - self.atoms[g[j]]) / self.exp_length))
      else:
        for i in range(0, num):
          for j in range(0, i):
            lc.append(np.linalg.norm(self.atoms[g[i]] - self.atoms[g[j]]))
      nr = len(lc)
      for i in range(0, nr):
        for j in range(0, i + 1): # added self-square here
          lc.append(lc[i] * lc[j])
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
    xx = mat_x[num - 1]
    ll = self.mat_ll
    r = []
    for k in range(0, xx.shape[0]):
      r.append([])
      if self.exp_length != 0:
        for i in range(0, xx.shape[1]):
          for j in range(0, i):
            r[k].append(exp(-ll[xx[k, i], xx[k, j]] / self.exp_length))
      else:
        for i in range(0, xx.shape[1]):
          for j in range(0, i):
            r[k].append(ll[xx[k, i], xx[k, j]])
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
  json_data = re.sub(r'//.*\n', '\n', json_data)
  return json.loads(json_data)

# write json summary
def write_summary(json_data, fn):
  fn = new_file_name(fn)
  f = open(fn, 'w')
  json.dump(json_data, f, indent=4)
  f.close()

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

# all permutation of atoms of a length list
# return matrix of indices
def all_per_new(n):
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

def get_length_new(n):
  m = n * (n - 1) / 2
  zz = np.zeros((2, m), dtype=int)
  mi = 0
  for i in range(0, n):
    for j in range(0, i):
      zz[0, mi] = i
      zz[1, mi] = j
      mi += 1
  return zz

def get_length_self_new(n):
  m = n * (n + 1) / 2
  zz = np.zeros((2, m), dtype=int)
  mi = 0
  for i in range(0, n):
    for j in range(0, i + 1):
      zz[0, mi] = i
      zz[1, mi] = j
      mi += 1
  return zz

def test_trans_data(x, y, num):
  n = len(x)
  gl = all_per_new(num)
  for i in range(0, n):
    xx = x[i, 0]
    yy = y[i]
    l = [0, 1]
    for g in gl:
      if ((x[i][1] - xx[g]) < 1E-10).all():
        l = [1, 0]
        break
    if not (l == yy).all():
      print (l == yy)

def ipsize_new(n, d_order=False):
  size = n * (n - 1) / 2
  if d_order: size = size + size * (size + 1) / 2
  return size

def trans_data_new(clus, n, num, typed, npi_network=None, d_order=False):
  sn = n
  if typed == 'npic':
    print ('prepare original coords array ...')
    xn = len(clus)
    gn = clus[0].n
    x = np.zeros((len(clus) * (gn + 1 - num), num, 3))
    for i in range(0, gn + 1 - num):
      for j in xrange(xn):
        clus[j].shuffle()
        x[i * xn + j] = clus[j].atoms[0:num]
    print ('prepare permutation array ...')
    pp = all_per_new(num)
    p = np.zeros((sn, 2, len(pp[0])), dtype=int)
    pn = len(pp)
    lend = len(pp[0])
    x_d = np.zeros((n, 2, ipsize_new(num, d_order)))
    for i in xrange(0, sn / 2):
      ind = random.randrange(pn)
      ind2 = random.randrange(pn)
      p[i][0] = pp[ind]
      while ind2 == ind: ind2 = random.randrange(pn)
      p[i][1] = pp[ind2]
    for i in xrange(sn / 2, sn):
      if i % (n / 100) == 0: print '{0} %'.format(i / (n / 100))
      p[i][0] = pp[random.randrange(pn)]
      pl = list(pp[random.randrange(pn)])
      while pl in pp:
        random.shuffle(pl)
      p[i][1] = pl
    yp = np.asarray([[1, 0]] * (sn / 2) + [[0, 1]] * ( sn - sn / 2))
    ip = range(0, sn)
    random.shuffle(ip)
    y_d = yp[ip]
    xp = p[ip]
    print ('apply permutation ...')
    xnn = x.shape[0]
    gl = get_length_new(num)
    gls = get_length_self_new(len(pp[0]))
    expl = clus[0].exp_length
    if expl == 0:
      for i in xrange(0, n):
        idx = random.randrange(0, xnn)
        xdr = np.linalg.norm(x[idx, gl[0]] - x[idx, gl[1]], axis=1)
        xdr = xdr[xp[i]]
        x_d[i, :, 0:lend] = xdr
        if d_order: x_d[i, :, lend:] = xdr[:, gls[0]] * xdr[:, gls[1]]
    else:
      for i in xrange(0, n):
        idx = random.randrange(0, xnn)
        xdr = np.exp(-np.linalg.norm(x[idx, gl[0]] - x[idx, gl[1]], axis=1) / expl)
        xdr = xdr[xp[i]]
        x_d[i, :, 0:lend] = xdr
        if d_order: x_d[i, :, lend:] = xdr[:, gls[0]] * xdr[:, gls[1]]
  return x_d, y_d

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
    idx = random.randrange(len(clus))
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
      idxg = random.randrange(2)
      clu = clus[idx]
      clu.shuffle(num)
      pll = clu.gen_per_ll(num)
      if idxg == 0:
        idx = random.randrange(len(pll))
        plx = [x for x in pll[idx]]
        random.shuffle(plx)
        while plx in pll:
          random.shuffle(plx)
        x_d.append([pll[0], plx])
        y_d.append([0, 1]) # different
      else:
        idx = random.randrange(len(pll) - 1) + 1
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

# find length max, min
# used when exp does not apply
def find_l_max_min(dat, min_max_ext_ratio):
  ratio = min_max_ext_ratio
  dmax = np.ma.max(dat[0])
  dmin = np.ma.min(dat[0])
  for d in dat[1:]:
    dmaxr = np.ma.max(d)
    dminr = np.ma.min(d)
    if dmaxr > dmax: dmax = dmaxr
    if dminr < dmin: dmin = dminr
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
  "tanhsig": layers.TanhSig, 
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
  opts = { "error": "mse", "step": ipdata["step"], 
    "verbose": False if typed == "npi" else True, 
    "batch_size": ipdata["batch_size"], "nesterov": True, 
    "momentum": ipdata["momentum"], "shuffle_data": True, 
    "show_epoch": ipdata["show_epoch"] }
  return algorithms.Momentum(x_layers, **opts)

# transfer weight and bias from neta to netb
def transfer_parameters(neta, netb):
  n = len(netb.layers)
  for i in range(n): 
    if isinstance(netb.layers[i], layers.ParameterBasedLayer):
      netb.layers[i].weight.set_value(neta.layers[i].weight.get_value())
      netb.layers[i].bias.set_value(neta.layers[i].bias.get_value())

# main program
if __name__ == "__main__":
  if len(sys.argv) < 1 + 1:
    print ('Need input file!')
  else:
    ip = read_input(sys.argv[1])
    ipdt = ip["data_files"]
    ippn = ip["npi_network"]
    ipft = ip["fit_network"]
    
    npic_network_name = ipdt["output_dir"] + "/npic_network.dill"
    npic_data_name = ipdt["output_dir"] + "/npic_data.dill"
    npic_test_name = ipdt["output_dir"] + "/npic_test.txt"
    npic_error_name = ipdt["output_dir"] + "/npic_error.txt"
    fit_network_name = ipdt["output_dir"] + "/fit_network.dill"
    fit_data_name = ipdt["output_dir"] + "/fit_data.dill"
    fit_test_name = ipdt["output_dir"] + "/fit_test.txt"
    fit_error_name = ipdt["output_dir"] + "/fit_error.txt"
    summary_name = ipdt["output_dir"] + "/summary.txt"
    
    if not os.path.exists(ipdt["output_dir"]):
      os.mkdir(ipdt["output_dir"])
    
    print ('load primary data ...')
    rcopts = {
      "ener": ipdt["list_file"], "xyz": ipdt["struct_file"], 
      "ecut": ipdt["energy_cut"], "expl": ipdt["exp_length"], 
      "traj": ipdt["trajectory_form"]
    }
    clus = read_cluster(**rcopts)
    dmax, dmin = find_max_min(clus, ipdt["min_max_ext_ratio"])
    ip["extra"] = {}
    ip["extra"]["energy_max"] = dmax
    ip["extra"]["energy_min"] = dmin
    random.shuffle(clus)
    nd = ipdt["degree_of_fitting"]
    
    for task in ip["task"]:
      ip["extra"][task] = {}
      # NPI comparing
      if task == "npic":
        print ('create network ...')
        if ippn["load_network"] != -1:
          if isinstance(ippn["load_network"], int):
            npic_net = load_data(name=npic_network_name, i=ippn["load_network"])
          else:
            npic_net = load_data(name=ippn["load_network"])
        else:
          npic_net = create_network(ippn, ipsize_new(nd, ipdt['second_order']), 'npic')
        
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
            x, y = trans_data_new(clus, ipdt["sample_number"][i], nd, typed="npic", 
              d_order=ipdt['second_order'])
            # x, y = trans_data(clus, ipdt["sample_number"][i], nd, typed="npic")
            npic_data += [x, y]
          if ipdt["scale_lengths"]:
            xmax, xmin = find_l_max_min(npic_data[0:6:2], ipdt["min_max_ext_ratio"])
            ip["extra"][task]["coord_max"] = xmax
            ip["extra"][task]["coord_min"] = xmin
            for i in range(0, 6, 2): npic_data[i] = trans_forward(npic_data[i], xmax, xmin)
        if ipdt["dump_data"]:
          print ('dump data ...')
          dump_data(name=npic_data_name, obj=npic_data)
        
        print ('input data shape: ', npic_data[0].shape)
        print ('output data shape: ', npic_data[1].shape)
        print ('first input data: \n', npic_data[0])
        print ('first output data: \n', npic_data[1])
        # test_trans_data(npic_data[0], npic_data[1], nd)
        print (npic_data[0].dtype, npic_data[1].dtype)
        
        if ippn["train_network"]:
          print ('train network ...')
          npic_net.train(*npic_data[0:4], epochs=ippn["epochs"])
        
        if ippn["test_network"]:
          print ('test network ...')
          ipd = npic_net.format_input_data(npic_data[4])
          nsqm = -1
          for i in range(len(npic_net.layers)):
            if isinstance(npic_net.layers[i], layer_dic[ippn["calibrate_method"]]):
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
            ft.write('%8s %10s %10s %15s\n' % ('id', 'standard', 'predict', 'calibrate'))
          for idx, std, pre, ind in zip(range(ntest), npic_std, npic_final, npic_ind):
            if std == pre: nr += 1
            if ippn["test_output"]:
              ft.write('%8d %10d %10d %15.8f\n' % (idx, std, pre, ind))
          if ippn["test_output"]: ft.close()
          print (nr * 100 / ntest, '%')
          if ippn["error_output"]:
            ft = open(new_file_name(npic_error_name), 'w')
            ft.write("# %d " % (nr * 100 / ntest, ) + '%\n')
            for i, j in zip(npic_net.errors, npic_net.validation_errors):
              ft.write('%15.8f %15.8f\n' % (i, j))
            ft.close()
        
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
          if ipft["load_npic_network"] != -1:
            fit_net = create_network(ipft, ippn["sizes"][-1], 'fit')
          else:
            fit_net = create_network(ipft, nd * (nd - 1) / 2, 'fit')
        
        print ('transfrom data ...')
        if ipdt["load_data"] != -1:
          if isinstance(ipdt["load_data"], int):
            fit_data = load_data(name=fit_data_name, i=ipdt["load_data"])
          else:
            fit_data = load_data(name=ipdt["load_data"])
        else:
          if ipft["load_npic_network"] != -1:
            print ('create npi network ...')
            if isinstance(ipft["load_npic_network"], int):
              npic_net = load_data(name=npic_network_name, i=ipft["load_npic_network"])
            else:
              npic_net = load_data(name=ippn["load_network"])
            npi_net = create_network(ippn, ipsize_new(nd, ipdt['second_order']), 'npi')
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
          if ipdt["scale_lengths"]:
            xmax, xmin = find_l_max_min(fit_data[0:6:2], ipdt["min_max_ext_ratio"])
            ip["extra"][task]["coord_max"] = xmax
            ip["extra"][task]["coord_min"] = xmin
            for i in range(0, 6, 2): fit_data[i] = trans_forward(fit_data[i], xmax, xmin)
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
          fit_pre = np.asarray([f[0] for f in fit_pre])
          fit_std = fit_data[5]
          fit_std = trans_backward(fit_std, dmax, dmin)
          fit_pre = trans_backward(fit_pre, dmax, dmin)
          ntest = len(fit_pre)
          nr = 0.0
          if ipft["test_output"]:
            ft = open(new_file_name(fit_test_name), 'w')
            ft.write('%8s %15s %15s %15s\n' % ('id', 'standard', 'predict', 'error'))
          for idx, std, pre in zip(range(ntest), fit_std, fit_pre):
            nr += (std - pre) ** 2
            if ipft["test_output"]:
              ft.write('%8d %15.8f %15.8f %15.8f\n' % (idx, std, pre, abs(std - pre)))
          if ipft["test_output"]:
            ft.close()
          print ('%15.8f\n' % (sqrt(nr / ntest) * httoev), ' eV')
          if ipft["error_output"]:
            ft = open(new_file_name(fit_error_name), 'w')
            ft.write('# %15.8f eV\n' % (sqrt(nr / ntest) * httoev))
            for i, j in zip(fit_net.errors, fit_net.validation_errors):
              ft.write('%15.8f %15.8f\n' % (i, j))
            ft.close()
        
        if ipft["dump_network"]:
          print ('dump network ...')
          dump_data(name=fit_network_name, obj=fit_net)
    
    if ipdt["dump_summary"]:
      print ('dump summary ...')
      write_summary(ip, summary_name)
