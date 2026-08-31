from pathlib import Path

import numpy as np
import pytest
import torch

from rl_trainer.agent_trainer.agent_models.agent_model_factory import AgentModelFactory
from rl_trainer.agent_trainer.agent_models.ppo_agent_model import PPOAgentModel
from rl_trainer.agent_trainer.agent_models.random_agent_model import RandomAgentModel
from rl_trainer.agent_trainer.argument_parser import (
    SUPPORTED_AGENT_MODELS,
    SUPPORTED_GAMES,
    create_argument_parser,
)
from rl_trainer.agent_trainer.game_environments import GameEnvironmentFactory

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_argument_parser_defaults_and_choices():
    parser = create_argument_parser()
    defaults = parser.parse_args([])

    assert defaults.game_name == "running"
    assert defaults.train_model == "ppo"
    assert defaults.enemy_model == "random"

    with pytest.raises(SystemExit):
        parser.parse_args(["--game_name", "unsupported"])


def test_factories_reject_unregistered_types():
    game_factory = GameEnvironmentFactory(SUPPORTED_GAMES)
    model_factory = AgentModelFactory(SUPPORTED_AGENT_MODELS)

    with pytest.raises(ValueError, match="Unsupported game"):
        game_factory.create_game_environment("unsupported")
    with pytest.raises(ValueError, match="Unsupported agent"):
        model_factory.create_agent_model("unsupported")


def test_random_agent_action_is_in_declared_ranges():
    force, angle = RandomAgentModel().get_action(np.zeros((40, 40)))

    assert -100 <= force <= 200
    assert -30 <= angle <= 30


def test_ppo_model_loads_curated_checkpoint_and_infers():
    model = PPOAgentModel()
    checkpoint = REPOSITORY_ROOT / "artifacts" / "ppo-vs-random-episode-9900"
    model.load(checkpoint)

    action, probability = model.get_action(torch.zeros((1, 40, 40)), train=False)

    assert 0 <= action < model.ACTION_SPACE
    assert 0.0 <= probability <= 1.0
    model.writer.close()
