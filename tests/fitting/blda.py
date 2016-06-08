
# Bond Length Distribution Algorithm
# For generating random structures

import numpy as np
import numpy.core.defchararray as npch
from math import *
import re, os, sys, json
from formod.at_sort import at_comp

class Cluster(object):
  def __init__(self, n):
    self.n = n
    self.atoms = np.zeros((n, 3), dtype=np.float64)
    self.elems = np.array([''] * n)
    self.energy = 0.0
    self.eps = np.finfo(dtype=np.float64).eps * 100
  
  # sample uniformly in the unit sphere
  def random_direction(self):
    d = np.random.random(size=2)
    phi, theta = 2 * np.pi * d[0], np.arccos(2 * d[1] - 1)
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), 
      np.cos(theta)])
  
  # create cluster using first-order bond length distribution method
  def create_simple(self, elems, mean, sigma):
    self.atoms[0] = 0.0
    self.elems = elems
    # sample uniformly in the unit sphere
    d = np.random.random(size=(self.n, 2))
    phi, theta = 2 * np.pi * d[:, 0], np.arccos(2 * d[:, 1] - 1)
    d = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), 
      np.cos(theta)]).T
    for i in range(1, self.n):
      x = 0.0
      for j in range(0, i):
        m = elems[i] + '-' + elems[j]
        l = np.random.normal(mean[m][0], sigma[m][0])
        b = -2 * np.dot(self.atoms[j], d[i])
        c = np.square(self.atoms[j]).sum() - l ** 2
        delta = b**2 - 4*c + self.eps
        if delta > 0.0:
          xx = (-b + np.sqrt(delta)) / 2
          if xx > x: x = xx
      self.atoms[i] = x * d[i]
      self.atoms[0:i+1] -= self.atoms[i] / (i + 1)
  
  # create cluster using second-order bond length distribution method
  def create(self, elems, mean, sigma):
    self.elems = elems
    indx = range(self.n)
    np.random.shuffle(indx)
    elems[indx] = elems[:]
    self.atoms[0] = 0.0
    m = elems[0] + '-' + elems[1]
    l = np.random.normal(mean[m][0], sigma[m][0])
    self.atoms[1] = self.random_direction() * l
    self.atoms[0:2] -= self.atoms[1] / 2
    for i in range(2, self.n):
      while True:
        shu = range(i)
        np.random.shuffle(shu)
        ix, iy = shu[0:2]
        mx = elems[i] + '-' + elems[ix]
        my = elems[i] + '-' + elems[iy]
        lx = np.random.normal(mean[mx][0], sigma[mx][0])
        if my != mx:
          ly = np.random.normal(mean[my][0], sigma[my][0])
        else:
          ly = 0
          while ly < lx:
            ly = np.random.normal(mean[my][1], sigma[my][1])
        surfn = self.atoms[iy] - self.atoms[ix]
        lz = np.linalg.norm(surfn)
        if lz > lx + ly: continue
        lxx = lz * lx / (lx + ly)
        lr = np.sqrt(lx ** 2 - lxx ** 2)
        cent = surfn * lx / (lx + ly) + self.atoms[ix]
        surfn = surfn / np.linalg.norm(surfn)
        ok = False
        for j in range(10):
          dd = self.random_direction()
          ra = dd - np.dot(dd, surfn) * surfn
          ra = cent + ra * lr / np.linalg.norm(ra)
          okr = True
          for k in shu[2:]:
            if np.linalg.norm(self.atoms[k] - ra) < lx:
              okr = False
              break
          if okr:
            ok = True
            break
        if ok:
          self.atoms[i] = ra
          self.atoms[0:i+1] -= self.atoms[i] / (i + 1)
          break
    self.atoms = self.atoms[indx]
  
  # return the position of the center
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  # will change the coordinates
  def center(self):
    self.atoms = self.atoms - self.get_center()
  
  # get original data for measuring the bond length distribution
  def get_dist(self):
    ell = []
    eln = []
    for i in range(self.n):
      if self.elems[i] in eln:
        ell[eln.index(self.elems[i])].append(i)
      else:
        eln.append(self.elems[i])
        ell.append([i])
    idel = np.argsort(np.array([len(l) for l in ell]))[::-1]
    ell = [ell[i] for i in idel]
    eln = [eln[i] for i in idel]
    res = {}
    for i in range(len(ell)):
      for j in range(i + 1):
        x = np.linalg.norm(self.atoms[ell[i]].reshape((len(ell[i]), 1, 3)) - 
          self.atoms[ell[j]].reshape((1, len(ell[j]), 3)), axis=2)
        y = eln[i] + '-' + eln[j]
        yp = eln[j] + '-' + eln[i]
        x = np.sort(x, axis=1)
        if i == j: x = x[:, 1:]
        res[y], res[yp] = x, x
    return res
  
  # write coordinates to file
  def write_xyz(self, fn):
    f = open(fn, 'a')
    f.write('%d\nE = %15.8f\n' % (self.n, self.energy))
    for x, y in zip(self.elems, self.atoms):
      f.write('%5s%15.8f%15.8f%15.8f\n' % (x, y[0], y[1], y[2]))
    f.close()

# read xyz structures from one file
def read_clusters(fn, fnum):
  clul = []
  f = open(fn, 'r')
  fs = f.readlines()
  f.close()
  fs = [[g for g in f.replace('\n', '').split(' ') if len(g) != 0] for f in fs]
  cc = 0
  while cc < len(fs) and (len(clul) < fnum or fnum == -1):
    cn = int(fs[cc][0])
    clu = Cluster(cn)
    i = 0
    if len(fs[cc + 1]) == 3:
      clu.energy = float(fs[cc + 1][2])
    for f in fs[cc + 2:cc + 2 + cn]:
      clu.elems[i] = f[0]
      ar = np.asarray([float(g) for g in f[1:4]])
      clu.atoms[i] = ar
      i = i + 1
    clu.center()
    clul.append(clu)
    cc += 2 + cn
  print ('structs loaded: %d' % (len(clul), ))
  return clul

# elems list to number used by at_sort
def elem_num(elems):
  lel = []
  cel = []
  for i in elems:
    if i not in cel:
      cel.append(i)
      lel.append(len(cel))
    else:
      j = cel.index(i)
      lel.append(j + 1)
  return lel

rxa = r'^\s*([A-Z]*[a-z]+)\s*([0-9]+)(.*)$'
rxb = r'^\s*\(\s*([^\)]+)\s*\)\s*([0-9]+)(.*)$'
rxc = r'^\s*([A-Z]*[a-z]+)(.*)$'
rxd = r'^\s*$'

# cluster name solve
def elem_char(name):
  dc = []
  dct = []
  i = 0
  while len(re.findall(rxd, name)) == 0:
    ra = re.findall(rxa, name)
    if len(ra) == 0:
      ra = re.findall(rxb, name)
    if len(ra) != 0:
      ne = ra[0][0][:1].upper() + ra[0][0][1:]
      dc += [ne] * int(ra[0][1])
      dct += [ne, ra[0][1]]
      name = ra[0][2]
    else:
      ra = re.findall(rxc, name)
      ne = ra[0][:1].upper() + ra[0][1:]
      dc += [ne]
      dct += [ne]
      name = ra[1]
  return np.array(dc), ' '.join(dct)

# avoid overwritting
def new_file_name(x):
  i = 0
  y = x + '.' + str(i)
  while os.path.isfile(y):
    i += 1
    y = x + '.' + str(i)
  return y

# read json input
def read_json(fn, i=None):
  if not i is None: fn += '.' + str(i)
  json_data = open(fn).read()
  json_data = re.sub(r'//.*\n', '\n', json_data)
  return json.loads(json_data)

# write json summary
def write_json(json_data, fn):
  fn = new_file_name(fn)
  f = open(fn, 'w')
  json.dump(json_data, f, indent=4)
  f.close()

# main program
if __name__ == "__main__":
  if len(sys.argv) < 1 + 1:
    print ('Need input file!')
  else:
    ip = read_json(sys.argv[1])
    
    structs_name = ip["data"]["output_dir"] + "/structs.xyz"
    output_name = ip["data"]["output_dir"] + "/structs_out.txt"
    
    if not os.path.exists(ip["data"]["output_dir"]):
      os.mkdir(ip["data"]["output_dir"])
    
    for task in ip["task"]:
      if task == "create":
        elems, et = elem_char(ip["cluster"]["name"])
        clnum = ip["cluster"]["number"]
        opts = { "elems": elems, "mean": ip["stat"]["mean"], 
          "sigma": ip["stat"]["sigma"] }
        clul = []
        fname = new_file_name(structs_name)
        dmax = ip["filtering"]["max_diff"]
        dmax_rep = ip["filtering"]["max_diff_report"]
        order = ip["cluster"]["order"]
        eles = elem_num(elems)
        ne = len(eles)
        nsim = 0
        out_str = []
        out_str.append("# struct name: %s" % (et, ))
        print (out_str[-1])
        out_str.append("# struct write to file: %s" % (fname, ))
        print (out_str[-1])
        # generate
        while len(clul) < clnum:
          c = Cluster(elems.shape[0])
          if order == 2: c.create(**opts)
          elif order == 1: c.create_simple(**opts)
          ok = True
          for i in clul:
            v, _ = at_comp(i.atoms, c.atoms, ne, eles, dmax_rep)
            if v < dmax:
              ok = False
              break
          if ok:
            clul.append(c)
            out_str.append(" struct # %5d / %5d: [S = %5d]" % (len(clul), clnum, nsim))
            print (out_str[-1])
            c.write_xyz(fname)
            nsim = 0
          else:
            nsim += 1
        # test
        cg = [c.get_dist() for c in clul]
        keys = cg[0].keys()
        for k in keys:
          out_str.append(" bond length: %s" % (k, ))
          print (out_str[-1])
          cgr = [ci[k] for ci in cg]
          dist = np.concatenate(cgr, axis=0)[:, 0:2]
          for i in range(dist.shape[1]):
            out_str.append(" -- # %d: %15.6f (%15.6f)" % (i, 
              np.average(dist[:, i]), np.std(dist[:, i])))
            print (out_str[-1])
          
        ft = open(new_file_name(output_name), 'w')
        for i in out_str: ft.write(i + '\n')
        ft.close()
