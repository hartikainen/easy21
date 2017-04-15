import random
from functools import reduce
from operator import mul
from collections import Iterable

from environment import ACTIONS


def elementwise_mul(*args):
  scalars = tuple( a for a in args if not isinstance(a, Iterable) )
  iterables = tuple( a for a in args if isinstance(a, Iterable) )
  result = [reduce(mul, I+scalars, 1) for I in zip(*iterables)]


def dot(x, y):
  assert len(x) == len(y)
  return sum(a * b for a,b in zip(x,y))


def MSE(A, B):
  mse = sum(
    (A[s][a] - B[s][a]) ** 2
    for s in set(A.keys()) | set(B.keys())
    for a in ['HIT', 'STICK']
  ) / (len(set(A.keys()) | set(B.keys())))
  return mse


def get_step_size(N_s_a):
  return 1.0/N_s_a


N_0 = 100
def get_epsilon(state_value):
  return N_0 / (N_0 + state_value)


def policy_wrapper(Q, N):
  def policy(state):
    eps = get_epsilon(sum(N[state].values()))

    if random.random() < (1 - eps):
      action, _ = max(Q[state].items(), key=lambda s: s[1])
    else:
      action = random.choice(ACTIONS)
    return action
  return policy
