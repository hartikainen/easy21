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
def plot_V(Q, save=None, fig_idx=0):
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


def plot_learning_curve(learning_curves, save=None, agent_args={}, fig_idx=2):
  fig = plt.figure(fig_idx)

  plt.title("Mean-squared error vs. 'true' Q values against episode number")
  plt.ylabel(r'$\frac{1}{|S||A|}\sum_{s,a}{(Q(s,a) - Q^{*}(s,a))^2}$', size=18)
  plt.xlabel(r'$episode$', size=18)

  colors = iter(cm.rainbow(np.linspace(0, 1, len(learning_curves))))
  for lmbd, D in learning_curves.items():
    X, Y = zip(*D)
    plt.plot(X, Y, label="lambda={:.1f}".format(lmbd),
             linewidth=1.0, color=next(colors))

  plt.legend()

  if save is not None:
    plt.savefig(save, format='pdf', transparent=True)
  else:
    plt.show()

  plt.clf()

def plot_pg_rewards(mean_rewards, save=None, fig_idx=3):
  fig = plt.figure(fig_idx)

  plt.title("Average results for Policy Gradient")
  plt.ylabel(r'$reward$', size=18)
  plt.xlabel(r'$episode$', size=18)

  Y = mean_rewards
  X = range(1, len(Y)+1)

  plt.plot(X, Y, linewidth=1.0)

  if save is not None:
    plt.savefig(save, format='pdf', transparent=True)
  else:
    plt.show()

  plt.clf()
