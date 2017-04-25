import numpy as np

from utils import epsilon_greedy_policy
from vis import plot_V
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

FEATS_SHAPE = tuple(
  len(CUBOID_INTERVALS[key]) for key in ("dealer", "player", "action")
)


def phi(state, action=None):
  if state == TERMINAL_STATE: return 0

  dealer, player = state

  state_features = np.array([
    (di[0] <= dealer <= di[1]) and (pi[0] <= player <= pi[1])
    for di in CUBOID_INTERVALS['dealer']
    for pi in CUBOID_INTERVALS['player']
  ]).astype(int).reshape(FEATS_SHAPE[:2])

  if action is None: return state_features

  features = np.zeros(FEATS_SHAPE)
  for i, ai in enumerate(CUBOID_INTERVALS['action']):
    if action in ai:
      features[:, :, i] = state_features

  return features.astype(int)


class FunctionApproximationAgent:
  def __init__(self, env, num_episodes=1000,
               gamma=GAMMA, lmbd=LAMBDA,
               epsilon=EPSILON, alpha=ALPHA,
               **kwargs):
    self.num_episodes = num_episodes
    self.env = env

    self.gamma = gamma
    self.lmbd = lmbd
    self.epsilon = epsilon
    self.alpha = alpha

    self.reset()


  def reset(self):
    self.Q = np.zeros(STATE_SPACE_SHAPE)
    self.w  = (np.random.rand(*FEATS_SHAPE) - 0.5) * 0.001


  def policy(self, state):
    if state == TERMINAL_STATE:
      return 0.0, None

    if np.random.rand() < (1 - self.epsilon):
      Qhat, action = max(
        # same as dotproduct in our case
        ((np.sum(phi(state, a) * self.w), a) for a in ACTIONS),
        key=lambda x: x[0]
      )
    else:
      action = np.random.choice(ACTIONS)
      Qhat = np.sum(phi(state, action) * self.w)

    return Qhat, action


  def expand_Q(self):
    Q = np.zeros(STATE_SPACE_SHAPE)

    for dealer in DEALER_RANGE:
      for player in PLAYER_RANGE:
        for action in ACTIONS:
          state = (dealer, player)
          feats = phi(state, action)
          Q[dealer-1, player-1][action] = np.sum(feats * self.w)

    return Q


  def learn(self):
    env = self.env
    N = np.zeros(STATE_SPACE_SHAPE)

    for episode in range(1, self.num_episodes+1):
      env.reset()
      state1 = env.observe()
      E = np.zeros(FEATS_SHAPE)

      while state1 != TERMINAL_STATE:
        Qhat1, action1 = self.policy(state1)
        state2, reward = env.step(action1)
        Qhat2, action2 = self.policy(state2)

        feats1 = phi(state1, action1)
        feats2 = phi(state2, action2)


        grad_w_Qhat1 = feats1

        delta = reward + self.gamma * Qhat2 - Qhat1
        E = self.gamma * self.lmbd * E + grad_w_Qhat1
        dw = self.alpha * delta * E

        self.w += dw
        state1 = state2

    Q = self.expand_Q()
    return Q
