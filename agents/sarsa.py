import numpy as np
import pickle

from utils import epsilon_greedy_policy, mse
from vis import plot_V
from environment import Easy21Env, TERMINAL_STATE, STATE_SPACE_SHAPE


GAMMA = 1.0
LAMBDA = 0

class SarsaAgent:
  def __init__(self, env, num_episodes=1000,
               gamma=GAMMA, lmbd=LAMBDA,
               save_error_history=False,
               **kwargs):
    self.num_episodes = num_episodes
    self.env = env
    self.gamma = gamma
    self.lmbd = lmbd

    self.save_error_history = save_error_history
    if self.save_error_history:
      with open("./Q_opt.pkl", "rb") as f:
        self.opt_Q = pickle.load(f)

    self.reset()


  def reset(self):
    self.Q = np.zeros(STATE_SPACE_SHAPE)

    if self.save_error_history:
      self.error_history = []


  def learn(self):
    env = self.env
    Q = self.Q
    N = np.zeros(STATE_SPACE_SHAPE)

    for episode in range(1, self.num_episodes+1):
      env.reset()
      state1 = env.observe()
      E = np.zeros(STATE_SPACE_SHAPE) # eligibility traces

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

        N[idx1] += 1
        E[idx1] += 1

        alpha = 1.0 / N[idx1]
        delta = reward + self.gamma * Q2 - Q1
        E *= self.gamma * self.lmbd
        Q += alpha * delta * E

        state1 = state2

      if self.save_error_history:
        self.error_history.append((episode, mse(self.Q, self.opt_Q)))

    return Q
