
import formod.muller as ml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

print ml.__doc__

x = np.linspace(-1.4, 1, 100)
y = np.linspace(-0.4, 2, 100)
z = np.asarray([[ml.xqsd_eval([xi, xj]) for xi in x] for xj in y])

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(x, y, z, levels=np.linspace(-200,100,30), colors='k')
plt.show()