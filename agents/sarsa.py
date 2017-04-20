import numpy as np

from utils import epsilon_greedy_policy
from vis import plot_Q
from environment import Easy21Env, TERMINAL_STATE, STATE_SPACE_SHAPE


GAMMA = 1.0
LAMBDA = 0

class SarsaAgent:
  def __init__(self, env, num_episodes=1000,
               gamma=GAMMA, lmbd=LAMBDA,
               **kwargs):
    self.num_episodes = num_episodes
    self.env = env
    self.gamma = gamma
    self.lmbd = lmbd
    self.reset()


  def reset(self):
    self.Q = np.zeros(STATE_SPACE_SHAPE)


  def learn(self):
    env = self.env
    Q = self.Q
    N = np.zeros(STATE_SPACE_SHAPE)

    for episode in range(1, self.num_episodes+1):
      env.reset()
      state1 = env.observe()
      # eligibility traces
      E = np.zeros(STATE_SPACE_SHAPE)

      while state1 != TERMINAL_STATE:
        action1 = epsilon_greedy_policy(Q, N, state1)
        state2, reward = env.step(action1)

        dealer1, player1 = state1
        idx1 = (dealer1-1, player1-1, action1)
        Q1 = Q[idx1]

        if state2 == TERMINAL_STATE:
          Q2 = 0.0
        else:
          action2 = epsilon_greedy_policy(Q, N, state2)
          dealer2, player2 = state2
          idx2 = (dealer2-1, player2-1, action2)
          Q2 = Q[idx2]

        delta = reward + self.gamma * (Q2 - Q1)

        N[idx1] += 1
        E[idx1] += 1

        alpha = 1.0 / N[idx1]
        Q += alpha * delta * E
        E *= GAMMA * self.lmbd

        state1 = state2

      if self.lmbd == 0 or self.lmbd == 1:
        # TODO: save the episodes
        pass
    return Q
