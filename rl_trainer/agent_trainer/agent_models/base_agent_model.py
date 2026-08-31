import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn

from rl_trainer.agent_trainer.util import Transition

logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseAgentModel(metaclass=ABCMeta):
    CLIP_PARAM = 0.2
    MAX_GRAD_NORM = 0.5
    PPO_UPDATE_TIME = 10
    BUFFER_CAPACITY = 1000
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 0.0001

    ACTION_SPACE = 36
    STATE_SPACE = 1600

    def __init__(self):
        self.transitions = []

    @property
    @abstractmethod
    def actor_nn(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def critic_nn(self) -> nn.Module:
        pass

    def store_transition(self, transition: Transition):
        self.transitions.append(transition)

    @abstractmethod
    def update_result(self, is_win: bool):
        pass

    @abstractmethod
    def get_action(self, state: np.array) -> Tuple[Tuple[float, float], int]:
        pass

    def save(self, save_file_path: str):
        logging.debug(f"Saving model", extra={'file_path': save_file_path})
        model_actor_path = os.path.join(save_file_path, "actor.pth")
        model_critic_path = os.path.join(save_file_path, "critic.pth")

        torch.save(self.actor_nn.state_dict(), model_actor_path)
        torch.save(self.critic_nn.state_dict(), model_critic_path)
        logging.debug(f"Finished saving model", extra={'file_path': save_file_path})

    def load(self, load_file_path: str):
        logging.debug(f"Loading model", extra={'file_path': load_file_path})
        actor_path = os.path.join(load_file_path, "actor.pth")
        critic_path = os.path.join(load_file_path, "critic.pth")
        if os.path.exists(critic_path) and os.path.exists(actor_path):
            actor = torch.load(actor_path, map_location=device)
            critic = torch.load(critic_path, map_location=device)
            self.actor_nn.load_state_dict(actor)
            self.critic_nn.load_state_dict(critic)
        else:
            logging.error("Failed to load models, files don't exist")
            raise ValueError("Tried to load models from files that don't exist")
        logging.debug(f"Loaded model", extra={'file_path': load_file_path})
