
import formod.muller as ml
import formod.lbfgs as lb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

import neupy, theano
from neupy import algorithms, layers
from neupy import environment

xg = np.linspace(-1.4, 1, 100)
yg = np.linspace(-0.4, 2, 100)
z = np.asarray([[ml.xqsd_evald([xi, xj])[1] for xi in xg] for xj in yg])
zx = np.asarray([[ml.xqsd_eval([xi, xj]) for xi in xg] for xj in yg])

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

plt.contour(xg, yg, z, levels=np.linspace(-200,100,30), colors='k')
plt.savefig('xg1.pdf')
plt.show()

plt.contour(xg, yg, zx, levels=np.linspace(-300,300,60), colors='k')

task = lb.LBFGS([0.25,1.8], 2)
task.p.eval = ml.xqsd_eval
task.p.evald = ml.xqsd_evald
task.log_file = 1
task.opt()
print ('ok')
plt.hold('on')
print (len(task.traj))
x = [t[0] for t in task.traj]
y = [t[1] for t in task.traj]
plt.plot(x, y, 'r-')
plt.savefig('xf1.pdf')
plt.show()

net = network = algorithms.Momentum(
  [ layers.Softplus(2), layers.Softplus(100), layers.Softplus(50), layers.Softplus(10), layers.Output(1) ],
  # error='binary_crossentropy',
  error='mse',
  step=0.1,
  verbose=True,
  batch_size = 10,
  nesterov = True,
  momentum = 0.8, 
  shuffle_data=True,
  show_epoch = 5
)

def gen_data(n):
  x_data = zip(np.random.uniform(-1.4, 1, n), np.random.uniform(-0.4, 2, n))
  y_data = [ ml.xqsd_eval(list(x)) for x in x_data ]
  y_data = [ 1.0 if y > 100 else (0.0 if y < -200 else (y + 200) / 300) for y in y_data ]
  return x_data, y_data

x_train, y_train = gen_data(50000)
x_test, y_test = gen_data(5000)

net.train(x_train, y_train, x_test, y_test, epochs=100)

xx = net.variables.network_input
xf = net.variables.prediction_func
xuf = theano.function([xx], xf)
xgx = theano.tensor.grad(xf[0][0], xx)
xgf = theano.function([xx], xgx)

zr = np.asarray([[xgf(np.array([xi, xj]).reshape((1, 2)))[0][1] for xi in xg] for xj in yg])
zr = [ ( z + 1) / 2 * 300 - 200 for z in zr]
zrf = np.asarray([[xuf(np.array([xi, xj]).reshape((1, 2)))[0][0] for xi in xg] for xj in yg])
zrf = [ z * 300 - 200 for z in zrf]

plt.contour(xg, yg, zr, levels=np.linspace(-200,100,60), colors='k')
plt.savefig('xg2.pdf')
plt.show()

plt.contour(xg, yg, zrf, levels=np.linspace(-200,100,30), colors='k')

task = lb.LBFGS([0.25,1.8], 2)
task.p.eval = lambda x, xuf=xuf: xuf(np.array(x).reshape((1, 2)))[0][0]
task.p.evald = lambda x, xgf=xgf: xgf(np.array(x).reshape((1, 2)))[0]
task.log_file = 1
task.opt()
print ('ok')
plt.hold('on')
print (len(task.traj))
x = [t[0] for t in task.traj]
y = [t[1] for t in task.traj]
plt.plot(x, y, 'r-')
plt.savefig('xf2.pdf')
plt.show()