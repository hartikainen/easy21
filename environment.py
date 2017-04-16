from __future__ import division
import random

DECK = range(1, 11)
ACTIONS = ( HIT, STICK ) = ( "HIT", "STICK" )
TERMINAL_STATE = "TERMINAL"
COEFFS = { 'red': -1, 'black': 1 }
P = { 'red': 1/3, 'black': 2/3 }


def draw_card(color=None):
  value = random.choice(DECK)
  if (color is None):
    if random.random() < P["red"]:
      color = "red"
    else:
      color = "black"
  return { 'value': value, 'color': color }


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

  if a == HIT:
    card = draw_card()
    player += COEFFS[card['color']] * card['value']

    if bust(player):
      next_state, reward = TERMINAL_STATE, -1
    else:
      next_state, reward = (dealer, player), 0
  elif a == STICK:
    while 0 < dealer < 17:
      card = draw_card()
      dealer += COEFFS[card['color']] * card['value']

    next_state = TERMINAL_STATE
    if bust(dealer):
      reward = 1
    else:
      reward = int(player > dealer) - int(player < dealer)
  else:
    raise ValueError("Action not in action space")

  return next_state, reward
