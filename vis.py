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
  # surf = ax.plot_wireframe(X, Y, Z)

  return surf


from environment import DEALER_RANGE, PLAYER_RANGE
def plot_V(Q, save=None, fig_idx=1):
  V = np.max(Q, axis=2)
  X, Y = np.mgrid[DEALER_RANGE, PLAYER_RANGE]

  surf = create_surf_plot(X, Y, V)

  plt.title("V*")
  plt.ylabel('player sum', size=18)
  plt.xlabel('dealer', size=18)

  if save is not None:
    plt.savefig(save, format='pdf', transparent=True)
  else:
    plt.show()

  plt.clf()


def plot_lambda_mse(data, save=None, figure_idx=0):
  if len(data) == 2:
    X, Y = data
  else:
    X, Y = list(range(len(data))), data

  fig = plt.figure(figure_idx)
  plt.plot(X, Y, 'bo-')
  plt.title("Mean-squared error vs. 'true' Q values against episode number")
  plt.ylabel(r'$\sum_{s,a}{(Q(s,a) - Q^{*}(s,a))^2}$', size=18)
  plt.xlabel(r'$\lambda$', size=18)

  if save is not None:
    plt.savefig(save, format='pdf', transparent=True)
  else:
    plt.show()

  plt.clf()


def plot_learning_curve(errors, save=None, agent_args={}, fig_idx=2):
  fig = plt.figure(fig_idx)
  ax = fig.add_subplot(111)
  plot = ax.plot(range(len(errors)), errors, ".", linewidth=2.0)
  plt.title("Learning curve of mean-squared error against episode number "
            "{} agent, lambda={}".format(
              agent_args["agent_type"], agent_args["lmbd"]))
  plt.ylabel(r'$\frac{1}{|S||A|}\sum_{s,a}{(Q(s,a) - Q^{*}(s,a))^2}$', size=18)
  plt.xlabel(r'$episode$', size=18)

  if save is not None:
    plt.savefig(save, format='pdf', transparent=True)
  else:
    plt.show()

  plt.clf()
