import numpy as np

from utils import epsilon_greedy_policy
from environment import TERMINAL_STATE, STATE_SPACE_SHAPE

class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)


  def forward(self, X):
    if len(X.shape) == 1: X = X.reshape(-1, X.shape[0])

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = 1, len(X)

    z = np.dot(X, W1) + b1
    h1 = np.maximum(z, 0)
    scores = np.dot(h1, W2) + b2

    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sum_exp_scores

    cache = (probs, h1, z)

    return probs, cache


  def backward(self, X, y, cache):
    if len(X.shape) == 1: X = X.reshape(-1, X.shape[0])

    probs, h1, z = cache

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = 1, len(X)

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1.0
    dscores /= float(N)

    dW2 = np.dot(h1.T, dscores) + W2
    db2 = np.sum(dscores, axis=0)

    dh1 = np.dot(dscores, W2.T)

    dz = (z > 0.0) * dh1

    dW1 = np.dot(X.T, dz) + W1
    db1 = np.sum(dz, axis=0)

    grads = {
      'W2': dW2, 'b2': db2,
      'W1': dW1, 'b1': db1
    }

    return grads


  def update_params(self, grads):
    learning_rate = 1e-4

    for param in self.params.keys():
      self.params[param] -= learning_rate * grads[param]


GAMMA = 1.0
LAMBDA = 0


class PolicyGradientAgent:
  def __init__(self, env, num_episodes=1000,
               gamma=GAMMA, lmbd=LAMBDA,
               save_error_history=False,
               **kwargs):
    self.num_episodes = num_episodes
    self.env = env
    self.gamma = gamma
    self.lmbd = lmbd

    self.reset()


  def reset(self):
    self.reward_history = []


  def learn(self):
    env = self.env
    net = TwoLayerNet(2, 20, 2)

    tmp_rewards = []

    for episode in range(1, self.num_episodes+1):
      env.reset()
      state = env.observe()
      E = [] # experiences

      while state != TERMINAL_STATE:
        probs, cache = net.forward(state)
        action = np.argmax(probs)
        state_, reward = env.step(action)

        E.append([state, action, reward, cache])
        state = state_

      G = np.cumsum([e[2] for e in reversed(E)])[::-1]
      for (state, action, reward, probs), G_t in zip(E, G):
        grads = net.backward(state, action, cache)
        grads = { k: v * G_t for k, v in grads.items() }
        net.update_params(grads)

      tmp_rewards.append(sum(e[2] for e in E))

      if episode % 1000 == 0:
        self.reward_history.append(np.mean(tmp_rewards))
        tmp_rewards = []

    return self.reward_history
