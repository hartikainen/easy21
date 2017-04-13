from collections import Counter, defaultdict

from utils import get_step_size, policy_wrapper, MSE
from vis import plot_Q, plot_MSE
from environment import step, draw_card, TERMINAL_STATE, ACTIONS
from monte_carlo import monte_carlo


STICK, HIT = ACTIONS
GAMMA = 1.0
NUM_EPISODES = 1000


def sarsa(lmbd=0):
  Q = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  N = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
  policy = policy_wrapper(Q, N)

  for episode in range(1, NUM_EPISODES+1):
    E = defaultdict(lambda: Counter({HIT: 0, STICK: 0}))
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state)
      state_, reward = step(state, action)
      action_ = policy(state_)
      delta = reward + GAMMA * Q[state_][action_] - Q[state][action]

      N[state][action] += 1
      E[state][action] += 1

      for s in E:
        for a in E[s]:
          if N[s][a] == 0: continue
          alpha = get_step_size(N[s][a])
          Q[s][a] += alpha * delta * E[s][a]
          E[s][a] *= GAMMA * lmbd

      state = state_

  return Q

if __name__ == "__main__":
  print("running sarsa")
  Q_ = monte_carlo()

  errors = {}
  for i in range(11):
    l = 0.1 * i
    Q = sarsa(l)
    errors[l] = MSE(Q, Q_)

  errors_table = list(zip(*errors.items()))
  plot_MSE(errors_table)
