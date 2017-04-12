import random
from environment import ACTIONS


def get_step_size(N_s_a):
  return 1.0/N_s_a


N_0 = 100
def get_epsilon(state_value):
  return N_0 / (N_0 + state_value)


def policy_wrapper(Q, N):
  def policy(state):
    eps = get_epsilon(sum(N[state].values()))

    if random.random() < (1 - eps):
      action, _ = max(Q[state].items(), key=lambda s: s[1])
    else:
      action = random.choice(ACTIONS)
    return action
  return policy
