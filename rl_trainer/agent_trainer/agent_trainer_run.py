import logging
import os
import random
from collections import deque
from datetime import datetime
from typing import Tuple

import numpy as np
import torch

from rl_trainer.agent_trainer.agent_models.agent_model_factory import AgentModelFactory
from rl_trainer.agent_trainer.agent_models.base_agent_model import BaseAgentModel
from rl_trainer.agent_trainer.agent_models.random_agent_model import RandomAgentModel
from rl_trainer.agent_trainer.argument_parser import parse_args, AgentTrainerArgs, SUPPORTED_GAMES, \
    SUPPORTED_AGENT_MODELS
from rl_trainer.agent_trainer.game_environments import GameEnvironmentFactory
from rl_trainer.agent_trainer.util import Transition

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}  # dicretise action space

MAX_MAP_INDEX = 12
RUNNING_BATCH_SIZE = 50


class AgentTrainer:
    SAVE_DIR_BASE = f"agent_trainer_saves"
    RANDOM_SINGLETON = RandomAgentModel()

    def __init__(self, agent_trainer_args: AgentTrainerArgs):
        logger.debug("Initializing Agent Trainer")
        torch.manual_seed(agent_trainer_args.torch_seed)
        self.game_factory = GameEnvironmentFactory(SUPPORTED_GAMES)
        self.game_env = next(self.game_env_generator)

        agent_model_factory = AgentModelFactory(SUPPORTED_AGENT_MODELS)
        self.training_agent_model = agent_model_factory.create_agent_model(agent_trainer_args.train_model)
        if agent_trainer_args.load_train_model:
            self.training_agent_model.load(agent_trainer_args.train_model_path)
        self.enemy_agent_model = agent_model_factory.create_agent_model(agent_trainer_args.enemy_model)
        if agent_trainer_args.load_enemy_model:
            self.enemy_agent_model.load(agent_trainer_args.enemy_model_path)

        torch.manual_seed(agent_trainer_args.torch_seed)
        self.save_dir = f"{self.SAVE_DIR_BASE}{os.sep}" \
                        f"{agent_trainer_args.train_model}_vs_{agent_trainer_args.enemy_model}{os.sep}" \
                        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.max_episodes = args.max_episodes
        self.max_episode_len = args.episode_max_len
        self.save_interval = args.save_interval
        self.is_render_graphics = args.render_graphic

        print(f"Arguments for AgentTrainer are:\n{agent_trainer_args}")
        logger.debug("Finished Initializing Agent Trainer")

    def _get_enemy_action(self, enemy_agent: BaseAgentModel, state: np.array, random_prob: float = 0.2) -> \
            Tuple[Tuple[float, float], int]:
        """
        Randomize the enemy movement every few times
        """
        if random.random() < random_prob:
            return self.RANDOM_SINGLETON.get_action(state)
        else:
            return enemy_agent.get_action(state)

    def run_episode(self, train_agent_index: int = 0):
        state = self.game_env.reset()

        if self.is_render_graphics:
            self.game_env.render()
        if isinstance(state[train_agent_index], type({})):
            train_agent_obs, train_agent_energy = state[train_agent_index]['agent_obs'], \
                self.game_env.agent_list[train_agent_index].energy
            enemy_agent_obs, enemy_agent_energy = state[1 - train_agent_index]['agent_obs'], \
                self.game_env.agent_list[1 - train_agent_index].energy
        else:
            train_agent_obs, train_agent_energy = state[train_agent_index], \
                self.game_env.agent_list[train_agent_index].energy
            enemy_agent_obs, enemy_agent_energy = state[1 - train_agent_index], \
                self.game_env.agent_list[1 - train_agent_index].energy

        train_agent_obs = torch.Tensor(np.array(train_agent_obs))  # (40, 40) -> torch.Size([1, 40, 40])
        train_agent_obs = train_agent_obs.unsqueeze(0)
        Gt = 0
        for step_num in range(self.max_episode_len):
            enemy_action = self._get_enemy_action(self.enemy_agent_model, enemy_agent_obs)
            train_agent_action_index, action_prob = self.training_agent_model.get_action(train_agent_obs)
            train_agent_action = actions_map[train_agent_action_index]
            action = [enemy_action, train_agent_action] if train_agent_index == 1 else [train_agent_action,
                                                                                        enemy_action]

            next_state, reward, done, _ = self.game_env.step(action)

            if isinstance(next_state[train_agent_index], type({})):
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[train_agent_index]['agent_obs'], \
                    self.game_env.agent_list[train_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = \
                    next_state[1 - train_agent_index]['agent_obs'], \
                        self.game_env.agent_list[1 - train_agent_index].energy
            else:
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[train_agent_index], \
                    self.game_env.agent_list[train_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1 - train_agent_index], \
                    self.game_env.agent_list[1 - train_agent_index].energy

            next_obs_ctrl_agent = torch.Tensor(np.array(next_obs_ctrl_agent))  # (40, 40) -> torch.Size([1, 40, 40])
            next_obs_ctrl_agent = next_obs_ctrl_agent.unsqueeze(0)
            train_reward = self.game_env.get_shaped_reward(reward[train_agent_index], done)

            trans = Transition(train_agent_obs, train_agent_action_index, action_prob,
                               train_reward,
                               next_obs_ctrl_agent, done)
            self.training_agent_model.store_transition(trans)

            enemy_agent_obs, enemy_agent_energy = next_obs_oppo_agent, next_energy_oppo_agent
            train_agent_obs, train_agent_energy = next_obs_ctrl_agent, next_energy_ctrl_agent

            if self.is_render_graphics:
                self.game_env.render()

            Gt += reward[train_agent_index] if done else -1

            if done:
                return reward

    def train(self):
        train_win_record = deque(maxlen=100)
        enemy_win_record = deque(maxlen=100)
        train_agent_index = 0
        train_count = 0
        for episode_num in range(self.max_episodes):
            self.game_env = next(self.game_env_generator)
            episode_reward = self.run_episode(train_agent_index)

            is_train_win = episode_reward[train_agent_index] > episode_reward[1 - train_agent_index]
            is_enemy_win = episode_reward[train_agent_index] < episode_reward[1 - train_agent_index]
            train_win_record.append(is_train_win)
            enemy_win_record.append(is_enemy_win)
            print("Episode: ", episode_num, "controlled agent: ", train_agent_index,  # "; Episode Return: ", Gt,
                  "; win rate(controlled & opponent): ", '%.2f' % (sum(train_win_record) / len(train_win_record)),
                  '%.2f' % (sum(enemy_win_record) / len(enemy_win_record)), '; Trained episode:', train_count)
            if is_train_win:
                train_count += 1
            self.training_agent_model.update_result(is_train_win == 1)

            if episode_num % self.save_interval == 0:
                episode_save_dir = f"{self.save_dir}{os.sep}episode_{episode_num}"
                os.makedirs(f"{self.save_dir}{os.sep}episode_{episode_num}")
                self.training_agent_model.save(episode_save_dir)


if __name__ == "__main__":
    args = parse_args()  # TODO: switch to training configuration yaml
    agent_trainer = AgentTrainer(args)
    agent_trainer.train()
