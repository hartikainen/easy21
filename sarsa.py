from collections import Counter, defaultdict

from utils import get_step_size, policy_wrapper
from helpers import MSE, plot_Q
from environment import step, draw_card, TERMINAL_STATE, ACTIONS
from monte_carlo import monte_carlo


STICK, HIT = ACTIONS
GAMMA = 1.0
NUM_EPISODES = 1000


def sarsa(l=0):
  Q = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  N = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  policy = policy_wrapper(Q, N)

  for episode in range(1, NUM_EPISODES+1):
    E = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
    state = (draw_card()['value'], draw_card()['value'])
    action = policy(state)

    while state != TERMINAL_STATE:
      N[state][action] += 1
      state_, reward = step(state, action)
      action_ = policy(state_)
      delta = reward + GAMMA * Q[state_][action_] - Q[state][action]
      E[state][action] += 1

      for s in E:
        for a in E[s]:
          if N[s][a] == 0: continue
          Q[s][a] += get_step_size(N[s][a]) * delta * E[s][a]
          E[s][a] *= l

      state, action = state_, action_
      policy = policy_wrapper(Q, N)

  return Q

if __name__ == "__main__":
  print("running sarsa")
  Q_ = monte_carlo()

  print("lambda, MSE")
  for i in range(11):
    l = 0.1 * i
    Q = sarsa(l)
    print("{:^6}, {:^4}".format(l, MSE(Q, Q_)))
