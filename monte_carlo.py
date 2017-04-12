from collections import Counter, defaultdict

from utils import get_step_size, policy_wrapper
from helpers import plot_Q
from environment import step, draw_card, TERMINAL_STATE, ACTIONS


STICK, HIT = ACTIONS


def update_policy(N, Q, episode_N, reward):
  for s, actions in episode_N.items():
    for a, count in actions.items():
      N[s][a] += count
      Q[s][a] += get_step_size(N[s][a]) * (reward - Q[s][a])

  return N, Q, policy_wrapper(Q, N)

NUM_EPISODES = 100000
def monte_carlo():
  Q = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  N = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  policy = policy_wrapper(Q, N)

  for episode in range(1, NUM_EPISODES+1):
    episode_N = defaultdict(lambda: Counter())
    reward = 0
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state)
      new_state, new_reward = step(state, action)

      episode_N[state][action] += 1
      reward += new_reward

      state = new_state

    N, Q, policy = update_policy(N, Q, episode_N, reward)

  return Q

if __name__ == "__main__":
  print("running monte carlo")
  Q = monte_carlo()
  plot_Q(Q)
