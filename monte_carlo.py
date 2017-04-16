import numpy as np

from utils import get_step_size, policy_wrapper
from vis import plot_Q
from environment import (
  step, draw_card, TERMINAL_STATE, ACTIONS, STATE_SPACE_SHAPE
)


def update_policy(N, Q, episode_N, reward):
  N += episode_N
  alpha = get_step_size(N)
  Q += alpha * (reward - Q)

  return N, Q, policy_wrapper(Q, N)


NUM_EPISODES = 10000
def monte_carlo():
  Q = np.zeros(STATE_SPACE_SHAPE)
  N = np.zeros(STATE_SPACE_SHAPE)
  policy = policy_wrapper(Q, N)

  for episode in range(1, NUM_EPISODES+1):
    episode_N = np.zeros(STATE_SPACE_SHAPE)
    total_reward = 0.0
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state)

      state_, reward = step(state, action)

      dealer, player = state
      episode_N[dealer-1, player-1, action] += 1.0

      total_reward += reward

      state = state_

    N, Q, policy = update_policy(N, Q, episode_N, total_reward)

  return Q


if __name__ == "__main__":
  print("running monte carlo")
  Q = monte_carlo()
  import pickle
  with open("./Q_opt.pkl", "wb") as f:
    pickle.dump(Q, f)
    plot_Q(Q)
