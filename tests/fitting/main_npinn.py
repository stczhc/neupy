
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

from formod.lbfgs import LBFGS
from formod.at_sort import at_sort, at_comp

if _platform == 'darwin':
  theano.config.cxx = "/usr/local/bin/g++-5"
sys.setrecursionlimit(100000)

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
    self.label = None
    self.multi = 1
  
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
def read_input(fn, i=None):
  if not i is None: fn += '.' + str(i)
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

# avoid overwritting
def new_path_name(x):
  i = 0
  y = x + '.' + str(i)
  while os.path.exists(y):
    i += 1
    y = x + '.' + str(i)
  return y

def dump_data(name, obj):
  name = new_file_name(name)
  print ('dump data: ' + name)
  with open(name, 'wb') as f:
    dill.dump(obj, f)

def load_data(name, i=None):
  if not i is None: name += '.' + str(i)
  print ('load data: ' + name)
  with open(name, 'rb') as f:
    return dill.load(f)

# return a list of Cluster objects
def read_cluster(ener, xyz, ecut, expl, traj, ecol):
  max_energy = ecut
  f = open(ener, 'r')
  fs = f.readlines()
  f.close()
  lf = []
  for f in fs:
    g = f.replace('\n', '').split(' ')
    g = [h for h in g if len(h) != 0]
    if len(g) == 0: continue
    if g[0] == 'id': continue
    if len(g) >= ecol and ecol != -1:
      lf += [[g[0], float(g[ecol - 1])]]
    else:
      lf += [[g[0], 0.0]]
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
      clu.label = l[0]
      if clu.energy < max_energy or ecol == -1:
        clul.append(clu)
      cc += 2 + cn
  print 'structs loaded: ', len(clul)
  return clul

# write optimized clusters to xyz file
def write_cluster(fn, elems, atoms):
  fnx = new_path_name(os.path.dirname(fn))
  os.mkdir(fnx)
  fna = os.path.basename(fn)
  le = len(elems)
  for i in range(atoms.shape[0]):
    f = open(fnx + '/' + fna.replace('#', str(i)), 'w')
    f.write(str(le) + '\n\n')
    at = atoms[i].reshape(8, 3)
    for x, y in zip(elems, at):
      f.write('%5s%15.8f%15.8f%15.8f\n' % (x, y[0], y[1], y[2]))
    f.close()

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

# exclude similar clusters
def filter_cluster(clus, dmax, dmax_rep, method):
  gn = clus[0].n
  lx = [list(g) for g in itertools.permutations(range(0, gn), gn)]
  lx = np.asarray(lx)
  gl = get_length_new(gn)
  lena = len(clus)
  lend = len(gl[0])
  clur = []
  k = 0
  lastl = 0
  out_str = []
  if method == 'atomic_sort': clul = np.zeros((lena, lend))
  elif method == 'direct': clul = np.zeros((lena, 1, lend))
  elif method == 'atomic_compare': clul = np.zeros((lena, gn, 3))
  ne = 1
  eles = np.ones((gn, ))
  for i, cl in zip(xrange(lena), clus):
    if lena >= 10 and i % (lena / 10) == 0:
      print ('{0} %'.format(int(i / (lena / 100.0))))
    if method == 'atomic_sort': 
      xx = at_sort(cl.atoms, ne, eles)
    elif method == 'direct': 
      xx = cl.atoms[lx]
      xx = np.linalg.norm(xx[:, gl[0]] - xx[:, gl[1]], axis=2)
    elif method == 'atomic_compare':
      xx = cl.atoms
    sim = -1
    psmi = -1
    psm = 10.0
    for hl in range(k):
      hlg = (hl + lastl) % k
      if method == 'atomic_sort': v = np.amax(np.abs(clul[hlg] - xx))
      elif method == 'direct': v = np.amin(np.amax(np.abs(clul[hlg] - xx), axis=1))
      elif method == 'atomic_compare': v, _ = at_comp(xx, clul[hlg], ne, eles, dmax_rep)
      # print v
      if v < dmax:
        sim = hlg
        break
      elif v < psm: psm, psmi = v, hlg
    if sim == -1:
      out_str.append("# %8s new (%5d ) E = %15.6f (%8s : dmax = %10.5f )" % 
        (cl.label, k + 1, cl.energy, clur[psmi].label if psmi != -1 else '', psm))
      if method == 'direct': print (out_str[-1])
      if method == 'atomic_sort': clul[k] = xx
      elif method == 'direct': clul[k][0] = xx[0]
      elif method == 'atomic_compare': clul[k] = xx
      clur.append(cl)
      lastl = k
      k += 1
    else:
      out_str.append("# %8s (E = %15.6f ) -> %8s (E = %15.6f ) dmax = %10.5f" % 
        (cl.label, cl.energy, clur[sim].label, clur[sim].energy, v))
      if method == 'direct': print (out_str[-1])
      clur[sim].multi += 1
      lastl = sim
  return clur, out_str
  
def trans_data_new(clus, n, num, typed, npi_network=None, d_order=False):
  sn = n
  if typed == 'opt':
    print ('prepare coords array ...')
    x_d = np.array([c.atoms for c in clus])
    y_d = np.array([c.energy for c in clus])
  elif typed == 'fit':
    print ('prepare original coords array ...')
    xn = len(clus)
    gn = clus[0].n
    lx = [list(g) for g in itertools.combinations(range(0, gn), num)]
    gl = get_length_new(num)
    gls = get_length_self_new(len(gl[0]))
    lend = len(gl[0])
    x_d = np.zeros((n, len(lx), ipsize_new(num, d_order)))
    if not npi_network is None:
      x_dx = np.zeros((n, len(lx), npi_network.output_layer.size))
    y_d = np.zeros((n, ))
    x = np.zeros((len(clus) * (gn + 1 - num), len(lx), num, 3))
    expl = clus[0].exp_length
    for i in range(n):
      if i % (n / 10) == 0: print '{0} %'.format(i / (n / 100))
      ind = random.randrange(xn)
      cl = clus[ind]
      cl.shuffle()
      xx = cl.atoms[lx]
      if expl == 0:
        xx = np.linalg.norm(xx[:, gl[0]] - xx[:, gl[1]], axis=2)
      else:
        xx = np.exp(-np.linalg.norm(xx[:, gl[0]] - xx[:, gl[1]], axis=2) / expl)
      x_d[i, :, 0:lend] = xx
      if d_order: 
        x_d[i, :, lend:] = xx[:, gls[0]] * xx[:, gls[1]]
      if not npi_network is None:
        x_dx[i] = npi_network.predict(x_d[i])
      y_d[i] = cl.energy
    if not npi_network is None: x_d = x_dx
  elif typed == 'npic':
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
      if i % (n / 10) == 0: print '{0} %'.format(i / (n / 100))
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

# the following has some problem because 
# the length products are also swapped
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

# update network parameters
# used when training but with different parameters
def update_network(net, ipdata):
  opts = { "error": "mse", "step": ipdata["step"], 
    "batch_size": ipdata["batch_size"], "nesterov": True, 
    "momentum": ipdata["momentum"], "shuffle_data": True, 
    "show_epoch": ipdata["show_epoch"] }
  for k, v in opts.items():
    setattr(net, k, v)

# return eval and evald functions for optimization
def opt_funs(net, ipdata, expl, xmax, xmin, dmax, dmin):
  x = T.dmatrix()
  totn = ipdata["number_of_atoms"]
  seln = ipdata["degree_of_fitting"]
  ee = np.eye(totn, dtype=int)
  le = np.array([list(g) for g in itertools.combinations(range(0, totn), seln)])
  lena = le.shape[0]
  lenb = seln * (seln - 1) / 2
  lx = np.zeros((2, lenb), dtype=np.int)
  k = 0
  for i in range(0, seln):
    for j in range(0, i):
      lx[:, k] = i, j
      k += 1
  yl = le[:, lx.T]
  lenc = totn * (totn - 1) / 2
  lt = np.zeros((2, lenc), dtype=np.int)
  k = 0
  lti = np.zeros((totn, totn), dtype=np.int)
  for i in range(0, totn):
    for j in range(0, i):
      lt[:, k] = i, j
      lti[i, j] = k
      lti[j, i] = k
      k += 1
  mt = (x[lt[0]] - x[lt[1]]).norm(2, axis=1)
  if expl != 0: mt = np.exp(-mt / expl)
  if not xmax is None: mt = trans_forward(mt, xmax, xmin)
  ylt = lti[yl[:, :, 0], yl[:, :, 1]]
  y = mt[ylt]
  ylf = yl.reshape((lena * lenb, 2))
  yltf = lti[ylf[:, 0], ylf[:, 1]]
  mtd = []
  for i in range(lenc):
    if i % (lenc / 10) == 0: print '{0} %'.format(int(i / (lenc / 100.0)))
    mtd.append(theano.function([x], T.grad(mt[i], x).flatten()))
  xx = net.variables.network_input
  xf = net.variables.prediction_func[0][0]
  xf = trans_backward(xf, dmax, dmin)
  xuf = theano.function([xx], xf)
  xy = theano.function([x], y)
  opt_eval = (lambda x, xuf=xuf, xy=xy, lena=lena, lenb=lenb, 
    totn=totn: xuf(xy(x.reshape(totn, 3)).reshape((1, lena, lenb))))
  xyd = lambda x, mtd=mtd, yltf=yltf: np.array([di(x) for di in mtd])[yltf]
  xfd = T.grad(xf, xx)
  xufd = theano.function([xx], xfd)
  opt_evald = (lambda x, xyd=xyd, xufd=xufd, xy=xy, lena=lena, lenb=lenb, totn=totn:
    np.tensordot(xufd(xy(x.reshape(totn, 3)).reshape((1, lena, lenb)))
      .reshape((lena * lenb, )), xyd(x.reshape(totn, 3)), axes=(0, 0)))
  return opt_eval, opt_evald

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
    ipdt = ip["data_files"] if "data_files" in ip.keys() else None
    ippn = ip["npi_network"] if "npi_network" in ip.keys() else None
    ipft = ip["fit_network"] if "fit_network" in ip.keys() else None
    ipop = ip["optimization"] if "optimization" in ip.keys() else None
    ipfl = ip["filtering"] if "filtering" in ip.keys() else None
    
    npic_network_name = ipdt["output_dir"] + "/npic_network.dill"
    npic_data_name = ipdt["output_dir"] + "/npic_data.dill"
    npic_test_name = ipdt["output_dir"] + "/npic_test.txt"
    npic_error_name = ipdt["output_dir"] + "/npic_error.txt"
    fit_network_name = ipdt["output_dir"] + "/fit_network.dill"
    fit_data_name = ipdt["output_dir"] + "/fit_data.dill"
    fit_test_name = ipdt["output_dir"] + "/fit_test.txt"
    fit_error_name = ipdt["output_dir"] + "/fit_error.txt"
    opt_data_name = ipdt["output_dir"] + "/opt_data.dill"
    opt_test_name = ipdt["output_dir"] + "/opt_test.txt"
    opt_list_name = ipdt["output_dir"] + "/opt_list.txt"
    opt_structs_name = ipdt["output_dir"] + "/opt_structs/pos_#.xyz"
    fil_list_name = ipdt["output_dir"] + "/fil_list.txt"
    fil_corr_name = ipdt["output_dir"] + "/fil_corr.txt"
    fil_structs_name = ipdt["output_dir"] + "/fil_structs/pos_#.xyz"
    summary_name = ipdt["output_dir"] + "/summary.txt"
    
    if not os.path.exists(ipdt["output_dir"]):
      os.mkdir(ipdt["output_dir"])
    
    print ('load primary data ...')
    rcopts = {
      "ener": ipdt["list_file"], "xyz": ipdt["struct_file"], 
      "ecut": ipdt["energy_cut"], "expl": ipdt["exp_length"], 
      "traj": ipdt["trajectory_form"]
    }
    rcopts['ecol'] = ipdt["energy_column"] if "energy_column" in ipdt else 1
    clus = read_cluster(**rcopts)
    ip["extra"] = {}
    if ipdt["energy_column"] != -1:
      dmax, dmin = find_max_min(clus, ipdt["min_max_ext_ratio"])
      ip["extra"]["energy_max"] = dmax
      ip["extra"]["energy_min"] = dmin
    if not "filter" in ip["task"] and (ipop is None or ipop["shuffle_input"]):
      random.shuffle(clus)
    natom = ipdt["number_of_atoms"]
    nd = ipdt["degree_of_fitting"]
    
    for task in ip["task"]:
      ip["extra"][task] = {}
      
      # FILTER
      if task == "filter":
        print ('filter structures ...')
        clur, corr = filter_cluster(clus, ipfl["max_diff"], 
          dmax_rep=ipfl["max_diff_report"], method=ipfl["method"])
        atoms = np.zeros((len(clur), clur[0].n, 3))
        ft = open(new_file_name(fil_list_name), 'w')
        ft.write('%8s %8s %8s %15s\n' % ('id', 'old-id', 'multi', 'pre-energy'))
        for idx, r in zip(xrange(len(clur)), clur):
          ft.write('%8d %8s %8d %15.8f\n' % (idx, r.label, r.multi, r.energy))
          atoms[idx] = r.atoms
        ft.close()
        ft = open(new_file_name(fil_corr_name), 'w')
        for c in corr: ft.write(c + '\n')
        ft.close()
        write_cluster(fil_structs_name, clur[0].elems, atoms)
        
      # OPT
      if task == "opt":
        print ('load network ...')
        if ipop["load_network"] != -1:
          if isinstance(ipop["load_network"], int):
            fit_net = load_data(name=fit_network_name, i=ipop["load_network"])
          else:
            fit_net = load_data(name=ipop["load_network"])
        else:
          print ('Need fitted network!')
          exit(-1)
        
        print ('load summary ...')
        if ipop["load_summary"] != -1:
          if isinstance(ipop["load_summary"], int):
            smip = read_input(fn=summary_name, i=ipop["load_summary"])
          else:
            smip = read_input(fn=ipop["load_summary"])
        else:
          print ('Need fitting summary!')
          exit(-1)
        
        if ipdt["load_data"] != -1:
          if isinstance(ipdt["load_data"], int):
            opt_data = load_data(name=opt_data_name, i=ipdt["load_data"])
          else:
            opt_data = load_data(name=ipdt["load_data"])
        else:
          opt_data = trans_data_new(clus, 0, 0, typed="opt")
        
        if "coord_max" in smip["extra"]["fit"].keys():
          xmax = smip["extra"]["fit"]["coord_max"]
          xmin = smip["extra"]["fit"]["coord_min"]
        else:
          xmax = None
          xmin = None
        
        dmax, dmin = smip["extra"]["energy_max"], smip["extra"]["energy_min"]
        
        opt_eval, opt_evald = opt_funs(fit_net, ipdt, expl=ipdt["exp_length"], 
          xmax=xmax, xmin=xmin, dmax=dmax, dmin=dmin)
        
        print ('test network ...')
        ft = open(new_file_name(opt_test_name), 'w')
        ft.write('%8s %15s %15s %15s\n' % ('id', 'standard', 'predict', 'error'))
        nr = 0.0
        for idx, x, y in zip(range(0, len(opt_data[0])), opt_data[0], opt_data[1]):
          z = opt_eval(x.flatten())
          nr += (y - z) ** 2
          ft.write('%8d %15.8f %15.8f %15.8f\n' % (idx, y, z, abs(y - z)))
        ft.close()
        print ('%15.8f\n' % (sqrt(nr / len(opt_data[0])) * httoev), ' eV')
        
        print ('optimization ...')
        task = LBFGS(natom * 3)
        task.p.eval = opt_eval
        task.p.evald = opt_evald
        task.log_file = 0
        ft = open(new_file_name(opt_list_name), 'w')
        ft.write('%8s %15s %15s %15s %15s\n' % ('id', 'standard', 'original', 'final', 'change'))
        nopt = ipop["opt_number"]
        nopt = opt_data[0].shape[0] if nopt == -1 else min(opt_data[0].shape[0], nopt)
        finalx = np.zeros((nopt, natom * 3))
        for idx, x, y in zip(range(0, len(opt_data[0])), opt_data[0], opt_data[1]):
          if idx == nopt: break
          task.start(x.flatten())
          task.opt()
          fo, ff = task.traj[0][1], task.traj[-1][1]
          print ('# %8d: %15.8f -> %15.8f (%5d steps)' % (idx, fo, ff, len(task.traj)))
          ft.write('%8d %15.8f %15.8f %15.8f %15.8f\n' % (idx, y, fo, ff, ff - fo))
          finalx[idx] = task.x
        ft.close()
        write_cluster(opt_structs_name, clus[0].elems, finalx)
      
      # NPI comparing
      elif task == "npic":
        print ('create network ...')
        if ippn["load_network"] != -1:
          if isinstance(ippn["load_network"], int):
            npic_net = load_data(name=npic_network_name, i=ippn["load_network"])
          else:
            npic_net = load_data(name=ippn["load_network"])
          update_network(npic_net, ippn)
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
          update_network(fit_net, ipft)
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
            x, y = trans_data_new(clus, ipdt["sample_number"][i], nd, 
              typed="fit", npi_network=npi_net, d_order=ipdt['second_order'])
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
