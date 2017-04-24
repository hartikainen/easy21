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
  if color is None:
    colors, probs = zip(*COLOR_PROBS.items())
    color = np.random.choice(colors, p=probs)
  return { 'value': value, 'color': color }


def bust(x):
  return (x < 1 or 21 < x)


class Easy21Env:
  """ Easy21 environment

  Easy21 is a simple card game similar to Blackjack The rules of the game are as
  follows:

  - The game is played with an infinite deck of cards (i.e. cards are sampled
    with replacement)
  - Each draw from the deck results in a value between 1 and 10 (uniformly
    distributed) with a colour of red (probability 1/3) or black (probability
    2/3).
  - There are no aces or picture (face) cards in this game
  - At the start of the game both the player and the dealer draw one black
    card (fully observed)
  - Each turn the player may either stick or hit
  - If the player hits then she draws another card from the deck
  - If the player sticks she receives no further cards
  - The values of the player's cards are added (black cards) or subtracted (red
    cards)
  - If the player's sum exceeds 21, or becomes less than 1, then she "goes
    bust" and loses the game (reward -1)
  - If the player sticks then the dealer starts taking turns. The dealer always
    sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes
    bust, then the player wins; otherwise, the outcome - win (reward +1),
    lose (reward -1), or draw (reward 0) - is the player with the largest sum.
  """
  def __init__(self):
    self.reset()


  def reset(self, dealer=None, player=None):
    if dealer is None: dealer = draw_card()['value']
    self.dealer = dealer
    if player is None: player = draw_card()['value']
    self.player = player


  def observe(self):
    if not (self.dealer in DEALER_RANGE and self.player in PLAYER_RANGE):
      return TERMINAL_STATE
    return (self.dealer, self.player)


  def step(self, action):
    """ Step function

    Inputs:
    - action: hit or stick

    Returns:
    - next_state: a sample of the next state (which may be terminal if the
      game is finished)
    - reward
    """

    if action == HIT:
      card = draw_card()
      self.player += COLOR_COEFFS[card['color']] * card['value']

      if bust(self.player):
        next_state, reward = TERMINAL_STATE, -1
      else:
        next_state, reward = (self.dealer, self.player), 0
    elif action == STICK:
      while 0 < self.dealer < 17:
        card = draw_card()
        self.dealer += COLOR_COEFFS[card['color']] * card['value']

      next_state = TERMINAL_STATE
      if bust(self.dealer):
        reward = 1
      else:
        reward = int(self.player > self.dealer) - int(self.player < self.dealer)
    else:
      raise ValueError("Action not in action space")

    return next_state, reward
