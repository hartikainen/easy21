from collections import Counter, defaultdict
import random

from utils import get_step_size, policy_wrapper, MSE, dot, elementwise_mul
from vis import plot_Q, plot_MSE
from environment import step, draw_card, TERMINAL_STATE, ACTIONS
from monte_carlo import monte_carlo

GAMMA = 1
EPSILON = 0.05
ALPHA = 0.01

CUBOID_INTERVALS = {
  "dealer": [[1, 4], [4, 7], [7, 10]],
  "player": [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]],
  "action": [["hit"], ["stick"]]
}

def theta(n, std=0.01):
  return [(random.random() - 0.5) * std for _ in range(n)]

def phi(s, a):
  dealer, player = s
  features = (
    [int(i[0] <= dealer <= i[1]) for i in range(CUBOID_INTERVALS['dealer'])] +
    [int(i[0] <= player <= i[1]) for i in range(CUBOID_INTERVALS['player'])] +
    [int(a in i) for i in range(CUBOID_INTERVALS['action'])]
  )


def get_Q(params):
  Q = defaultdict(lambda: Counter({ HIT: 0, STICK: 0 }))
  for dealer in range(10):
    for player in range(21):
      s = (dealer, player)
      for a in ACTIONS:
        feats = phi(s, a)
        Q[s][a] = dot(feats, theta)

  return Q


NUM_EPISODES = 1000
def linear_function_approximation(lmbd):
  N = Counter({ f: 0 for f in range(3*6*2) })
  policy = None # TODO: need policy
  params = theta(3*6*2) # TODO: use variables for these

  for episode in range(1, NUM_EPISODES+1):
    E = Counter({ f: 0 for f in range(3*6*2) })
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state) # TODO: need policy
      state_, reward = step(state, action)

      feats = phi(state, action)
      for f in feats:
        if f == 1: E[f] += 1

      feats_, action_ = max(
        ((phi(_state, a), a) for a in ACTIONS),
        key=lambda x: dot(x[0], params)
      )

      delta = reward + GAMMA * dot(params, feats_) - dot(params, feats)

      for i in xrange(feats):
        N[i] += 1
        E *= GAMMA * lmbd

      dw = elementwise_mul(phi, E, ALPHA, delta)
      for i in xrange(len(dw)): params[i] += dw[i]

      state = state_

    Q = get_Q(params)

if __name__ == "__main__":
  print("running linear function approximation")
  Q_ = monte_carlo()

  errors = {}
  for i in range(11):
    lmbd = 0.1 * i
    Q = linear_function_approximation(lmbd)
    errors[lmbd] = MSE(Q, Q_)

  errors_table = list(zip(*errors.items()))
  plot_MSE(errors_table)
