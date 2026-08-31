import numpy as np

from agents.dueling_dqn.submission import my_controller as dueling_dqn_controller
from agents.game_classifier.submission import my_controller as classifier_controller


def assert_valid_action(action):
    assert len(action) == 2
    assert all(len(component) == 1 for component in action)
    assert -100 <= action[0][0] <= 200
    assert -30 <= action[1][0] <= 30


def test_dueling_dqn_submission_loads_weights_and_returns_action():
    observation = {"obs": {"agent_obs": np.zeros((40, 40), dtype=np.uint8)}}

    first = dueling_dqn_controller(observation, None, None)
    second = dueling_dqn_controller(observation, None, None)

    assert first == second
    assert_valid_action(first)


def test_classifier_submission_loads_all_weights_and_returns_action():
    observation = {"obs": {"agent_obs": np.zeros((40, 40), dtype=np.uint8)}}

    action = classifier_controller(observation, None, None)

    assert_valid_action(action)


def test_classifier_falls_back_for_non_image_observation():
    observation = {"obs": {"agent_obs": np.zeros((8,), dtype=np.float32)}}

    action = classifier_controller(observation, None, None)

    assert_valid_action(action)
