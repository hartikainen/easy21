from __future__ import division
import numpy as np

DECK = range(1, 11)
STICK_ACTION, HIT_ACTION = "STICK", "HIT"
TERMINAL_STATE = "TERMINAL"
P = { 'red': 1/3, 'black': 2/3 }


def draw_card(color=None):
  value = np.random.choice(DECK)
  if (color is None):
    color = np.random.choice(P.keys(), p=P.values())
  return {'value': value, 'color': color}


def bust(x):
  return (x < 1 or 21 < x)


def step(s, a):
  """ Step function

  Inputs:
  - s: state (dealer's first card 1-10 and the player's sum 1-21)
  - a: action (hit or stick)

  Returns:
  - next_state: a sample of the next state (which may be terminal if the
    game is finished)
  - reward
  """
  dealer, player = s

  if a == STICK_ACTION:
    while dealer < 17:
      dealer += draw_card()["value"]
    next_state = TERMINAL_STATE
    reward = int(player > dealer) - int(player < dealer)
  elif a == HIT_ACTION:
    card = draw_card()
    player += { "red":-1, "black":1 }[card['color']] * card['value']

    if (bust(player)):
      next_state, reward = TERMINAL_STATE, -1
    else:
      next_state, reward = (dealer, player), 0
  else:
    raise ValueError("Action not in action space")

  return next_state, reward
