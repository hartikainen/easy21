from collections import Counter, defaultdict
import numpy as np

from environment import (
  step, draw_card, TERMINAL_STATE, STICK_ACTION, HIT_ACTION
)

N_0 = 100


def get_step_size(N_s_a):
  return 1.0/N_s_a


def get_epsilon(state_value):
  return N_0 / (N_0 + state_value)


def policy_wrapper(Q, N):
  def policy(state):
    eps = get_epsilon(sum(N[state].values()))

    greedy_action, value = max(Q[state].items(), key=lambda s: s[1])
    random_action = STICK_ACTION if greedy_action == STICK_ACTION else HIT_ACTION

    action = np.random.choice([greedy_action, random_action], p=[1-eps, eps])
    return action
  return policy


def update_policy(N, Q, policy, episode_N, reward):
  for s, actions in episode_N.items():
    for a, count in actions.items():
      N[s][a] += count
      Q[s][a] += get_step_size(N[s][a]) * (reward - Q[s][a])

  return N, Q, policy_wrapper(Q, N)


def monte_carlo():
  Q = defaultdict(lambda: Counter({'HIT': 0, 'STICK': 0}))
  N = defaultdict(lambda: Counter({'HIT': 0, 'STICK': 0}))
  policy = policy_wrapper(Q, N)

  for episode in range(1, 1001):
    episode_N = defaultdict(lambda: Counter())
    reward = 0
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state)
      episode_N[state][action] += 1
      new_state, new_reward = step(state, action)
      state = new_state
      reward += new_reward

    N, Q, policy = update_policy(N, Q, policy, episode_N, reward)

  return Q

if __name__ == "__main__":
  print("running monte carlo")
  Q = monte_carlo()
