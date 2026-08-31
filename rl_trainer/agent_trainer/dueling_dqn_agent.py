# Dueling DQN Agent - 1 game at a time.
# Reward function (winning oriented):
# 1000 - Steps +  (if won + if lost -)Max(1000 - steps//5 + 10000, 0)
# action space discrete.


import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)

from olympics_engine.generator import create_scenario

from train.log_path import *
from train.algo.random import random_agent

from olympics_engine.scenario import table_hockey, football, Running_competition
from olympics_engine.agent import *


class DuelingDQN_Net(nn.Module):
    def __init__(self, obs_dim, num_actions, dropout):
        super(DuelingDQN_Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions)
        )

        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        advantage = self.advantage_stream(x)
        value = self.value_stream(x)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Calculate the softmax probabilities
        return F.softmax(q_values, dim=1)


class DuelingDQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained,
                 actions_number=2, is_train=True):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Dueling DQN Network
        self.dueling_dqn = DuelingDQN_Net(state_space, action_space, dropout).to(self.device)

        if self.pretrained:
            net_path = os.path.dirname(os.path.abspath(__file__)) + f"{os.sep}DuelingDQN.pt"
            self.dueling_dqn.load_state_dict(torch.load(net_path, map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dueling_dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained and is_train:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            self.ending_position = 0
            self.num_in_queue = 0

        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = float(done)  # Convert boolean to float
        self.ending_position = (self.ending_position + 1) % self.max_memory_size
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = torch.randint(0, self.num_in_queue, (self.memory_sample_size,))
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def get_eval_action(self, state):
        state_tensor = torch.Tensor([state])
        state_tensor = state_tensor.unsqueeze(1)
        with torch.no_grad():
            # Set the model to evaluation mode to disable learning
            self.dueling_dqn.eval()
            q_values = self.dueling_dqn(state_tensor.to(self.device))
            # Set the model back to training mode for future calls to act
            self.dueling_dqn.train()
            return torch.argmax(q_values).cpu()

    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:
            return torch.tensor(random.randrange(self.action_space))
        else:
            state = state.unsqueeze(1)
            return torch.argmax(self.dueling_dqn(state.to(self.device))).cpu()

    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.dueling_dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dueling_dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagation error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


NAME_TO_CLASS = {
    "running-competition": Running_competition,
    "football": football,
    "table-hockey": table_hockey,
}
EPISODE_NUM_PER_GAME = {game_class: 0 for game_class in NAME_TO_CLASS.values()}

MAX_MAP_INDEX = 4
BATCH_SIZE = 200


def training_games(game_scenarios: set = ('football', 'running-competition', 'table-hockey')):
    game_envs = []
    for scenario_name in game_scenarios:
        game_scenario = create_scenario(scenario_name)
        env_class = NAME_TO_CLASS[scenario_name]
        if scenario_name == 'running-competition':
            for map_id in range(1, MAX_MAP_INDEX):
                game_env = env_class(meta_map=game_scenario, map_id=map_id, vis=200, vis_clear=5,
                                     agent1_color='light red', agent2_color='blue')
                game_env.max_step = 400
                game_env.reset()
                game_envs.append(game_env)

        else:
            game_env = env_class(game_scenario)
            game_env.max_step = 400
            game_env.reset()
            game_envs.append(game_env)

    while True:
        training_env = random.choice(game_envs)
        print(f"~~~~~{training_env.__class__.__name__}~~~~~")
        for batch_index in range(BATCH_SIZE):
            EPISODE_NUM_PER_GAME[training_env.__class__] += 1
            yield training_env


def get_episodes_won_str(episodes_won: Dict[object, int]):
    episodes_won_str = ""
    for training_game, game_class in NAME_TO_CLASS.items():
        if EPISODE_NUM_PER_GAME[game_class] != 0:
            episodes_won_str += \
                f"Win Rate - {game_class}: {(episodes_won[game_class] / EPISODE_NUM_PER_GAME[game_class]) * 100:.2f}%, "
    return episodes_won_str


def reward_shaping(reward, steps, terminal, max_reward):
    shaped_reward = max_reward - steps

    if terminal:
        if reward > 0:  # Agent won the game
            shaped_reward += max(max_reward - (steps // 5) + 10000, 0)
        else:  # Agent lost the game
            shaped_reward -= max(max_reward - (steps // 5) + 10000, 0)

    return shaped_reward


def run(training_mode, pretrained, train_game: str, num_episodes=100000, exploration_max=1, is_render: bool = True):
    num_agents = 2
    ctrl_agent_index = 0  # controlled agent index

    # discrete action space
    actions_map = {
        0: [100, 0],  # N
        1: [100, 30],  # NE
        2: [100, -30],  # NW
        3: [-100, 0],  # S
        4: [-100, 30],  # SW
        5: [-100, -30],  # SE
    }

    print(f'Total agent number: {num_agents}')
    print(f'Agent control by the actor: {ctrl_agent_index}')

    load_model = False

    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
    observation_space = observation_space.shape

    # observation_space = shape=(40, 40, 1)
    action_space = len(actions_map)
    agent = DuelingDQNAgent(state_space=observation_space,
                            action_space=action_space,
                            max_memory_size=10000,
                            batch_size=128,
                            gamma=0.95,  # Increased from 0.90 to 0.95
                            lr=0.001,
                            dropout=0.2,
                            exploration_max=1.0,
                            exploration_min=0.02,
                            exploration_decay=0.995,  # Decreased from 0.99 to 0.995
                            pretrained=pretrained)

    opponent_agent = random_agent()  # we use random opponent agent here

    # Restart the environment for each episode
    num_episodes = num_episodes
    episodes_won = {game_class: 0 for game_class in NAME_TO_CLASS.values()}
    total_steps = 0
    scores_squared_sum = 0

    total_rewards = []
    max_reward = 1000

    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for ep_num, env in zip(tqdm(range(num_episodes)), training_games(game_scenarios={train_game})):

        episode_dir = f"./{start_time}/episode_{ep_num}"
        os.makedirs(episode_dir)

        state = env.reset()
        steps = 0

        if isinstance(state[ctrl_agent_index], type({})):
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'], env.agent_list[
                ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - ctrl_agent_index]['agent_obs'], env.agent_list[
                1 - ctrl_agent_index].energy
        else:
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - ctrl_agent_index], env.agent_list[1 - ctrl_agent_index].energy

        obs_ctrl_agent = torch.Tensor(np.array(obs_ctrl_agent))  # (40, 40) -> torch.Size([1, 40, 40])
        obs_ctrl_agent = obs_ctrl_agent.unsqueeze(0)
        # obs_ctrl_agent = torch.Tensor([obs_ctrl_agent])
        while True:
            if is_render:
                env.render()
            action = agent.act(obs_ctrl_agent)
            action_opponent = [0, 0]

            action_digit = action.data.tolist()
            action_ctrl = actions_map[action_digit]

            action_combined = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl,
                                                                                            action_opponent]
            steps += 1

            next_state, reward, terminal, info = env.step(action_combined)

            if isinstance(next_state[ctrl_agent_index], type({})):
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'], env.agent_list[
                    ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1 - ctrl_agent_index]['agent_obs'], \
                    env.agent_list[1 - ctrl_agent_index].energy
            else:
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[
                    ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1 - ctrl_agent_index], env.agent_list[
                    1 - ctrl_agent_index].energy

            next_obs_ctrl_agent = torch.Tensor(np.array(next_obs_ctrl_agent))  # (40, 40) -> torch.Size([1, 40, 40])
            next_obs_ctrl_agent = next_obs_ctrl_agent.unsqueeze(0)
            # next_obs_ctrl_agent = next_obs_ctrl_agent.transpose((1, 0, 2))
            # next_obs_ctrl_agent = torch.Tensor([next_obs_ctrl_agent])
            # Apply new reward function
            total_reward = reward_shaping(reward[ctrl_agent_index], steps, terminal, max_reward)

            if terminal:
                if reward[ctrl_agent_index] > reward[1 - ctrl_agent_index]:  # if agent won
                    episodes_won[env.__class__] += 1  # increment episodes won

            reward_ctrl = torch.Tensor([total_reward])

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(obs_ctrl_agent, action, reward_ctrl, next_obs_ctrl_agent, terminal)
                agent.experience_replay()

            state = next_state

            obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
            obs_ctrl_agent, energy_ctrl_agent = next_obs_ctrl_agent, next_energy_ctrl_agent

            if terminal:
                break

        total_rewards.append(total_reward)
        total_steps += steps  # increment total steps
        scores_squared_sum += total_reward ** 2  # add square of reward for std calculation

        if ep_num != 0 and ep_num % 100 == 0:
            std_dev = np.sqrt(
                max((scores_squared_sum / ep_num) - (np.mean(total_rewards) ** 2), np.finfo(float).eps))

            print(f"\nEp: {ep_num + 1}, Last Score: {total_rewards[-1]:.2f}, Avg Score: {np.mean(total_rewards):.2f}, "
                  f"Score Std Dev: {std_dev:.2f}, {get_episodes_won_str(episodes_won)}"
                  f"Avg Step: {total_steps / ep_num:.2f}. ")

            for game_class in NAME_TO_CLASS.values():
                EPISODE_NUM_PER_GAME[game_class] = 0
                episodes_won[game_class] = 0

            if training_mode:
                torch.save(agent.dueling_dqn.state_dict(), f"{episode_dir}/DuelingDQN.pt")
                torch.save(agent.STATE_MEM, f"{episode_dir}/STATE_MEM.pt")
                torch.save(agent.ACTION_MEM, f"{episode_dir}/ACTION_MEM.pt")
                torch.save(agent.REWARD_MEM, f"{episode_dir}/REWARD_MEM.pt")
                torch.save(agent.STATE2_MEM, f"{episode_dir}/STATE2_MEM.pt")
                torch.save(agent.DONE_MEM, f"{episode_dir}/DONE_MEM.pt")

    std_dev = np.sqrt(
        max((scores_squared_sum / ep_num) - (np.mean(total_rewards) ** 2), np.finfo(float).eps))

    print(f"\nEp: {ep_num + 1}, Last Score: {total_rewards[-1]:.2f}, Avg Score: {np.mean(total_rewards):.2f}, "
          f"Score Std Dev: {std_dev:.2f}, "
          f"{get_episodes_won_str(episodes_won)}"
          f"Avg Step: {total_steps / ep_num:.2f}. ")


# for training
if __name__ == "__main__":
    run(training_mode=True, pretrained=False, train_game="table-hockey", is_render=False)
