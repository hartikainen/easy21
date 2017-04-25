import numpy as np

from utils import epsilon_greedy_policy, mse
from vis import plot_V
from environment import Easy21Env, TERMINAL_STATE, STATE_SPACE_SHAPE


class MonteCarloAgent:
  def __init__(self, env, num_episodes=1000, **kwargs):
    self.num_episodes = num_episodes
    self.env = env
    self.reset()


  def reset(self):
    self.Q = np.zeros(STATE_SPACE_SHAPE)


  def learn(self):
    env = self.env
    Q = self.Q
    N = np.zeros(STATE_SPACE_SHAPE)

    for episode in range(1, self.num_episodes+1):
      env.reset()
      state = env.observe()
      E = [] # experience from the episode

      while state != TERMINAL_STATE:
        action = epsilon_greedy_policy(Q, N, state)
        state_, reward = env.step(action)

        E.append([state, action, reward])
        state = state_

      for (dealer, player), action, reward in E:
        idx = dealer-1, player-1, action
        N[idx] += 1
        alpha = 1.0 / N[idx]
        Q[idx] += alpha * (reward - Q[idx])

    return Q
