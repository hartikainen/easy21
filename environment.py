from __future__ import division
import numpy as np

DECK = range(1, 11)
ACTIONS = (HIT, STICK) = (0, 1)

DEALER_RANGE = range(1, 11)
PLAYER_RANGE = range(1, 22)
STATE_SPACE_SHAPE = (len(DEALER_RANGE), len(PLAYER_RANGE), len(ACTIONS))

TERMINAL_STATE = "TERMINAL"
COLOR_PROBS = { 'red': 1/3, 'black': 2/3 }
COLOR_COEFFS = { 'red': -1, 'black': 1 }


def draw_card(color=None):
  value = np.random.choice(DECK)
  if (color is None):
    colors, probs = zip(*COLOR_PROBS.items())
    color = np.random.choice(colors, p=probs)
  return { 'value': value, 'color': color }


def bust(x):
  return (x < 1 or 21 < x)


def step(state, action):
  """ Step function

  Inputs:
  - state: dealer's first card 1-10 and the player's sum 1-21
  - action: hit or stick

  Returns:
  - next_state: a sample of the next state (which may be terminal if the
    game is finished)
  - reward
  """
  dealer, player = state

  if action == HIT:
    card = draw_card()
    player += COLOR_COEFFS[card['color']] * card['value']

    if bust(player):
      next_state, reward = TERMINAL_STATE, -1
    else:
      next_state, reward = (dealer, player), 0
  elif action == STICK:
    while 0 < dealer < 17:
      card = draw_card()
      dealer += COLOR_COEFFS[card['color']] * card['value']

    next_state = TERMINAL_STATE
    if bust(dealer):
      reward = 1
    else:
      reward = int(player > dealer) - int(player < dealer)
  else:
    raise ValueError("Action not in action space")

  return next_state, reward
