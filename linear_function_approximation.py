from collections import Counter, defaultdict

from utils import get_step_size, policy_wrapper, MSE
from vis import plot_Q, plot_MSE
from environment import step, draw_card, TERMINAL_STATE, ACTIONS
from monte_carlo import monte_carlo

GAMMA = 1
THETA = None
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

NUM_EPISODES = 1000
def linear_function_approximation(lmbd):
  Q = defaultdict(lambda: Counter({ HIT: random.random() - 0.5,
                                    STICK: random.random() - 0.5 }))
  N = defaultdict(lambda: Counter({ HIT: random.random() - 0.5,
                                    STICK: random.random() - 0.5 }))
  policy = policy_wrapper(Q, N)
  params = theta(3*6*2) # TODO: use variables for these

  for episode in range(1, NUM_EPISODES+1):
    E = defaultdict(lambda: Counter({ HIT: 0, STICK: 0 }))
    state = (draw_card()['value'], draw_card()['value'])

    while state != TERMINAL_STATE:
      action = policy(state)
      state_, reward = step(state, action)
      action_ = policy(state_)

      feats = phi(state, action)
      feats_ = phi(state_, action_)

      # delta = reward + GAMMA * dot(feats_, theta) - dot(feats, theta)

if __name__ == "__main__":
  print("running linear function approximation")
  Q_ = monte_carlo()

  errors = {}
  for i in range(11):
    lmbd = 0.1 * i
    Q = sarsa(lmbd)
    errors[lmbd] = MSE(Q, Q_)

  errors_table = list(zip(*errors.items()))
  plot_MSE(errors_table)
