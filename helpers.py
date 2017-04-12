import matplotlib
matplotlib.use("TkAgg")

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import numpy as np
from pprint import pprint


def MSE(A, B):
  mse = sum(
    (A[s][a] - B[s][a]) ** 2
    for s in set(A.keys()) | set(B.keys())
    for a in ['HIT', 'STICK']
  ) / (len(A.keys()) + len(B.keys()))
  return mse


def create_surf_plot(X, Y, Z, fig_idx=1):
  fig = plt.figure(fig_idx)
  ax = fig.add_subplot(111, projection="3d")

  # X, Y = np.meshgrid(X, Y)
  surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)

  plt.show()


def plot_Q(Q):
  data = sorted([
    (dealer, player, max(action_value.values()))
    for (dealer, player), action_value in Q.items()
  ], key=lambda d: (d[0], d[1]))
  X, Y, Z = zip(*data)
  create_surf_plot(X, Y, Z)
