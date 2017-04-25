import numpy as np
from functools import reduce
from operator import mul
from collections import Iterable

from environment import ACTIONS


def mse(A, B):
  return np.sum((A - B) ** 2) / np.size(A)


def get_step_size(N):
  non_zeros = np.where(N > 0)
  steps = np.zeros_like(N)
  steps[non_zeros] = 1.0 / N[non_zeros]

  return steps


N_0 = 100
def get_epsilon(N):
  return N_0 / (N_0 + N)


def epsilon_greedy_policy(Q, N, state):
  dealer, player = state
  epsilon = get_epsilon(np.sum(N[dealer-1, player-1, :]))
  if np.random.rand() < (1 - epsilon):
    action = np.argmax(Q[dealer-1, player-1, :])
  else:
    action = np.random.choice(ACTIONS)
  return action


def policy_wrapper(Q, N):
  def policy(state):
    dealer, player = state
    assert(0 < dealer < 11 and 0 < player < 22)
    eps = get_epsilon(np.sum(N[dealer-1, player-1, :]))

    if np.random.rand() < (1 - eps):
      action = np.argmax(Q[dealer-1, player-1, :])
    else:
      action = np.random.choice(ACTIONS)
    return action
  return policy
