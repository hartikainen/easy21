from collections import Counter, defaultdict
import numpy as np

from utils import get_step_size, policy_wrapper, MSE
from vis import plot_Q, plot_MSE
from environment import step, draw_card, TERMINAL_STATE, ACTIONS
from monte_carlo import monte_carlo

HIT, STICK = ACTIONS

GAMMA = 1
EPSILON = 0.05
ALPHA = 0.01

CUBOID_INTERVALS = {
  "dealer": ((1, 4), (4, 7), (7, 10)),
  "player": ((1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)),
  "action": (("hit",), ("stick",))
}

FEATS_SHAPE = tuple(len(s) for s in CUBOID_INTERVALS.values())


def phi(s, a=None):
  if s == TERMINAL_STATE: return 0

  dealer, player = s

  state_features = np.array([
    (di[0] <= dealer <= di[1]) and (pi[0] <= player <= pi[1])
    for di in CUBOID_INTERVALS['dealer']
    for pi in CUBOID_INTERVALS['player']
  ]).astype(int).reshape(FEATS_SHAPE[:2])

  if a is None:
    return state_features

  features = np.zeros(FEATS_SHAPE)
  for i, ai in enumerate(CUBOID_INTERVALS['action']):
    if a in ai:
      features[:, :, i] = state_features

  return features


EPSILON = 0.05
def policy(s, w):
  if s == TERMINAL_STATE:
    return 0.0, None

  if np.random.rand() < (1 - EPSILON):
    Q, action = max(
      # same as dotproduct in our case
      ((np.sum(phi(s, a) * w), a) for a in ACTIONS),
      key=lambda x: x[0]
    )
  else:
    action = np.random.choice(ACTIONS)
    Q = np.sum(phi(s, action) * w)
  return Q, action


def expand_Q(w):
  Q = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))

  for dealer in range(10):
    for player in range(21):
      for ai, action in enumerate(ACTIONS):
        state = (dealer, player)
        feats = phi(state,action)
        Q[state][action] = np.sum(feats * w)

  return Q

NUM_EPISODES = 1000
def linear_function_approximation(lmbd):
  N = np.zeros(FEATS_SHAPE)
  w = (np.random.rand(*FEATS_SHAPE) - 0.5) * 0.01

  for episode in range(1, NUM_EPISODES+1):
    E = np.zeros(FEATS_SHAPE)
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      Qhat, action = policy(state, w)
      state_, reward = step(state, action)
      Qhat_, action_ = policy(state_, w)

      feats = phi(state, action)
      feats_ = phi(state_, action_)

      N[np.where(feats == 1)] += 1

      Qhat, Qhat_ = feats * w, feats_ * w

      delta = reward + GAMMA * Qhat_ - Qhat
      grad_w_Qhat = feats

      E = GAMMA * lmbd * E + grad_w_Qhat
      dw = ALPHA * delta * E

      w += dw
      state = state_

  Q = expand_Q(w)
  return Q


if __name__ == "__main__":
  print("running linear function approximation")
  Q_ = monte_carlo()

  print("lambda, error")

  errors = {}
  for i in range(11):
    lmbd = 0.1 * i
    Q = linear_function_approximation(lmbd)
    error = MSE(Q, Q_)
    errors[lmbd] = error

    print("{:.5}, {:.5}".format(lmbd, error))

  errors_table = list(zip(*errors.items()))
  plot_MSE(errors_table)
