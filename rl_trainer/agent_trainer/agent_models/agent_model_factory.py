from typing import Set

from rl_trainer.agent_trainer.agent_models.base_agent_model import BaseAgentModel
from rl_trainer.agent_trainer.agent_models.ppo_agent_model import PPOAgentModel
from rl_trainer.agent_trainer.agent_models.random_agent_model import RandomAgentModel

NAME_TO_CLASS = {
    "ppo": PPOAgentModel,
    "random": RandomAgentModel
}


class AgentModelFactory:
    def __init__(self, agent_models: Set[str]):
        if not agent_models.issubset(NAME_TO_CLASS.keys()):
            raise ValueError('Unsupported agent model registered to factory')
        self.agent_models = agent_models

    def create_agent_model(self, agent_model: str, *args) -> BaseAgentModel:
        if agent_model not in self.agent_models:
            raise ValueError('Unsupported agent model')
        agent_model_class = NAME_TO_CLASS[agent_model]
        return agent_model_class(*args)
