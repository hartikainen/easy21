import numpy as np

from utils import get_step_size, policy_wrapper, get_epsilon
from vis import plot_Q
from environment import (
  step, draw_card, TERMINAL_STATE, ACTIONS, STATE_SPACE_SHAPE
)


def update_policy(N, Q, episode_N, reward):
  N += episode_N
  alpha = get_step_size(N)
  Q += alpha * (reward - Q)

  return N, Q, policy_wrapper(Q, N)


NUM_EPISODES = 100000
def monte_carlo():
  Q = np.zeros(STATE_SPACE_SHAPE)
  N = np.zeros(STATE_SPACE_SHAPE)
  policy = policy_wrapper(Q, N)

  for episode in range(1, NUM_EPISODES+1):
    E = [] # experience from the episode
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      dealer, player = state

      # epsilon greedy policy
      epsilon = get_epsilon(np.sum(N[dealer-1, player-1, :]))
      if np.random.rand() < (1 - epsilon):
        action = np.argmax(Q[dealer-1, player-1, :])
      else:
        action = np.random.choice(ACTIONS)

      state, reward = step(state, action)

      E.append([dealer, player, action, reward])

    for dealer, player, action, reward in E:
      N[dealer-1, player-1, action] += 1
      alpha = 1.0 / N[dealer-1, player-1, action]
      Q[dealer-1, player-1, action] += alpha * (reward - Q[dealer-1, player-1, action])

  return Q


if __name__ == "__main__":
  print("running monte carlo")
  Q = monte_carlo()
  import pickle
  with open("./Q_opt.pkl", "wb") as f:
    pickle.dump(Q, f)
    plot_Q(Q)
