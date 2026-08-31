import random
from typing import Tuple

import numpy as np
from torch import nn

from rl_trainer.agent_trainer.agent_models.base_agent_model import BaseAgentModel


class RandomAgentModel(BaseAgentModel):
    def __init__(self):
        super().__init__()
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    @property
    def actor_nn(self) -> nn.Module:
        return nn.Module()

    @property
    def critic_nn(self) -> nn.Module:
        return nn.Module()

    def update_result(self, is_win: bool):
        pass

    def get_action(self, state: np.array) -> Tuple[Tuple[float, float], int]:  # TODO: fix to be uniform
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [force, angle]
