from unittest.mock import patch
from nose.tools import assert_equal

from environment import (
  Easy21Env, TERMINAL_STATE, ACTIONS
)

HIT, STICK = ACTIONS

def mock_draw_card(result):
  def fn(color=None):
    return result
  return fn


class TestEnvironment():
  def setUp(self):
    self.env = Easy21Env()


  def tearDown(self):
    self.env = None

  def test_step_player_should_bust_if_sum_exceeds_max(self):
    CARD = { 'value': 8, 'color': "black" }
    card_mock = mock_draw_card(CARD)
    PLAYER_START = 0
    self.env.reset(dealer=5, player=PLAYER_START)

    with patch("environment.draw_card", card_mock):
      state = self.env.observe()
      for i in range(3):
        player = state[1]
        assert_equal(self.env.player, PLAYER_START + i * CARD["value"])
        state, reward = self.env.step(HIT)

      assert_equal(state, TERMINAL_STATE)
      assert_equal(reward, -1)

  def test_step_player_should_bust_if_sum_below_min(self):
    CARD = { 'value': 8, 'color': "red" }
    card_mock = mock_draw_card(CARD)
    PLAYER_START = 10
    self.env.reset(dealer=5, player=PLAYER_START)

    with patch("environment.draw_card", card_mock):
      state = self.env.observe()
      for i in range(2):
        assert_equal(self.env.player, PLAYER_START - i * CARD["value"])
        state, reward = self.env.step(HIT)

      assert_equal(state, TERMINAL_STATE)
      assert_equal(reward, -1)

  def test_step_dealer_finishes_between_17_21(self):
    CARD = { 'value': 8, 'color': "black" }
    pass
