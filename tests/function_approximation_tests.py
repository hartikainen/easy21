from nose.tools import assert_equal
import glob
import json
import os

from environment import ACTIONS
from agents.function_approximation import phi

HIT, STICK = ACTIONS
NAME_TO_ACTION = {
  "hit": HIT,
  "stick": STICK
}
ACTION_TO_NAME = {
  v: k for k, v in NAME_TO_ACTION.items()
}

FEATURE_PATH_TEMPLATE = "./tests/fixtures/feature-{}.json"

def write_fixture(filepath, result):
  with open(filepath, "w") as f:
    json.dump(result, f, separators=(',', ': '), sort_keys=True, indent=2)


class TestFunctionApproximationAgent:
  def verify_features(self, filepath):
    with open(filepath) as f:
      test_case = json.load(f)

    state = test_case["state"]

    action = None
    if test_case.get("action", None) is not None:
      action = NAME_TO_ACTION[test_case["action"]]

    feats = phi(test_case["state"], action).tolist()
    expected = test_case["expected_feats"]

    if expected != feats and os.environ.get("TESTS_UPDATE", False):
      result = {
        "state": test_case["state"],
        "action": ACTION_TO_NAME.get(action, None),
        "expected_feats": feats
      }
      write_fixture(filepath, result)

    assert_equal(expected, feats)


  def test_features(self):
    features_path = FEATURE_PATH_TEMPLATE.format("*")
    for filepath in glob.glob(features_path):
      yield self.verify_features, filepath


  def create_feature_fixtures(self, state_actions):
    for state, action in state_actions:
      fixture = {
        "state": state,
        "action": action,
        "expected_feats": []
      }

      filename = "{}-{}".format(state[0], state[1])
      if action is not None:
        filename += "-{}".format(action)

      filepath = FEATURE_PATH_TEMPLATE.format(filename)
      write_fixture(filepath, fixture)
    pass
