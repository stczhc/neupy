# Bond Length Distribution Algorithm

import numpy as np
import numpy.core.defchararray as npch
from math import *
import matplotlib.pyplot as plt
fn = './data/blda_data/relaxed-1.xyz'
fn = './data/blda_data/relaxed-10-1.xyz'
# fn = './data/blda_data/new.xyz'
fnum = 100

class Cluster(object):
  def __init__(self, n):
    self.n = n
    self.atoms = np.zeros((n, 3), dtype=np.float64)
    self.elems = np.array([''] * n)
    self.energy = 0.0
    self.eps = np.finfo(dtype=np.float64).eps * 100
  
  # create cluster using bond length distribution method
  def create(self, elems, mean, sigma):
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
    xi, yi = self.get_dist()
    if np.any(xi[:, 1] > np.array([mean[y][1] + sigma[y][1] * 3 for y in yi[:, 1]])):
      self.create(elems, mean, sigma)
  
  # return the position of the center
  def get_center(self):
    return np.average(self.atoms, axis = 0)
  
  # will change the coordinates
  def center(self):
    self.atoms = self.atoms - self.get_center()
  
  def get_dist(self):
    x = np.linalg.norm(self.atoms.reshape((1, self.n, 3)) - 
      self.atoms.reshape((self.n, 1, 3)), axis=2)
    y = npch.add(npch.add(self.elems.reshape((1, self.n)), '-'), 
        self.elems.reshape((self.n, 1)))
    ind = np.argsort(x, axis=1)
    for i in range(x.shape[0]): x[i] = x[i, ind[i]]
    for i in range(y.shape[0]): y[i] = y[i, ind[i]]
    return x[:, 1:], y[:, 1:]
  
  def write_xyz(self, fn):
    f = open(fn, 'a')
    f.write('%d\nE = %15.8f\n' % (self.n, self.energy))
    for x, y in zip(self.elems, self.atoms):
      f.write('%5s%15.8f%15.8f%15.8f\n' % (x, y[0], y[1], y[2]))
    f.close()

clul = []
f = open(fn, 'r')
fs = f.readlines()
f.close()
fs = [[g for g in f.replace('\n', '').split(' ') if len(g) != 0] for f in fs]
cc = 0
while cc < len(fs) and len(clul) < fnum:
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
print 'structs loaded: ', len(clul)
cg = [c.get_dist()[0] for c in clul]
dist1 = np.concatenate(cg, axis=0)[:, 0]
dist2 = np.concatenate(cg, axis=0)[:, 1]
dist3 = np.concatenate(cg, axis=0)[:, 2]
dist4 = np.concatenate(cg, axis=0)[:, 3]
ener = np.array([c.energy +1193 for c in clul])
print np.average(dist1), np.std(dist1)
print np.average(dist2), np.std(dist2)
print np.average(ener), np.std(ener)

# n, bins, patches = plt.hist(dist2, 50, normed=True)
# plt.show()

# exit(0)

clul = []
for i in range(0, 100):
  print i
  c = Cluster(8)
  c.create(np.array(['Pt'] * 8), {'Pt-Pt': [2.500, 2.558] }, {'Pt-Pt': [0.042, 0.169] })
  c.write_xyz('./data/blda_data/new.xyz')
  clul.append(c)

cg = [c.get_dist()[0] for c in clul]
dist1 = np.concatenate(cg, axis=0)[:, 0]
dist2 = np.concatenate(cg, axis=0)[:, 1]
print np.average(dist1), np.std(dist1)
print np.average(dist2), np.std(dist2)
n, bins, patches = plt.hist(dist2, 50, normed=True)
plt.show()