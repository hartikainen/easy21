import matplotlib
matplotlib.use("TkAgg")

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

import numpy as np
from pprint import pprint


def create_surf_plot(X, Y, Z, fig_idx=1):
  fig = plt.figure(fig_idx)
  ax = fig.add_subplot(111, projection="3d")

  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False)

  plt.show()


from environment import DEALER_RANGE, PLAYER_RANGE
def plot_Q(Q):
  V = np.max(Q, axis=2)
  X, Y = np.mgrid[DEALER_RANGE, PLAYER_RANGE]

  create_surf_plot(X, Y, V)


def plot_MSE(data, figure_idx=0):
  if len(data) == 2:
    X, Y = data
  else:
    X, Y = list(range(len(data))), data

  fig = plt.figure(figure_idx)
  plt.plot(X, Y, 'bo-')
  plt.ylabel(r'$\sum_{s,a}{(Q(s,a) - Q^{*}(s,a))^2}$', size=20)
  plt.xlabel(r'$\lambda$', size=20)
  plt.show()
