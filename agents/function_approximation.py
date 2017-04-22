import numpy as np

from utils import epsilon_greedy_policy
from vis import plot_Q
from environment import (
  Easy21Env, TERMINAL_STATE, STATE_SPACE_SHAPE, ACTIONS,
  DEALER_RANGE, PLAYER_RANGE
)

HIT, STICK = ACTIONS


GAMMA = 1
LAMBDA = 0
EPSILON = 0.05
ALPHA = 0.01

CUBOID_INTERVALS = {
  "dealer": ((1, 4), (4, 7), (7, 10)),
  "player": ((1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)),
  "action": ((HIT,), (STICK,))
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

  return features.astype(int)


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
  Q = np.zeros(STATE_SPACE_SHAPE)

  for dealer in DEALER_RANGE:
    for player in PLAYER_RANGE:
      for ai, action in enumerate(ACTIONS):
        state = (dealer, player)
        feats = phi(state, action)
        Q[dealer-1, player-1][action] = np.sum(feats * w)

  return Q


class FunctionApproximationAgent:
  def __init__(self, env, num_episodes=1000,
               gamma=GAMMA, lmbd=LAMBDA,
               epsilon=EPSILON, alpha=ALPHA,
               **kwargs):
    self.num_episodes = num_episodes
    self.env = env
    self.gamma = gamma
    self.lmbd = lmbd
    self.epislon = epsilon
    self.alpha = alpha
    self.reset()


  def reset(self):
    self.Q = np.zeros(STATE_SPACE_SHAPE)


  def learn(self):
    env = self.env
    N = np.zeros(STATE_SPACE_SHAPE)
    w  = (np.random.rand(*FEATS_SHAPE) - 0.05) * 0.01

    for episode in range(1, self.num_episodes+1):
      E = np.zeros(FEATS_SHAPE)
      state1 = env.observe()

      while state1 != TERMINAL_STATE:
        Qhat1, action1 = policy(state1, w)
        state2, reward = env.step(action1)
        Qhat2, action2 = policy(state2, w)

        feats1 = phi(state1, action1)
        feats2 = phi(state2, action2)

        N[np.where(feats1 == 1)] += 1

        Qhat1, Qhat2 = feats1 * w, feats2 * w

        delta = reward + GAMMA * Qhat2 - Qhat1
        grad_w_Qhat1 = feats1

        E = GAMMA * self.lmbd * E + grad_w_Qhat1
        dw = ALPHA * delta * E

        w += dw
        state1 = state2

    Q = expand_Q(w)
    return Q
