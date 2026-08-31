# https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe

import random
import sys
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import matplotlib.pyplot as plt
import argparse

# from utils import discrete_to_continuous


import os
from pathlib import Path

from tqdm import tqdm

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)

from collections import deque, namedtuple
import random

from olympics_engine.generator import create_scenario
from env.chooseenv import make
from rl_trainer.log_path import *
# from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random import random_agent

from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition
from olympics_engine.agent import *


class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        out = self.fc(conv_out)
        return out


class DQNAgent:
    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, pretrained, actions_number=2):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DQN network
        self.dqn = DQNSolver(state_space, action_space).to(self.device)

        if self.pretrained:
            self.dqn.load_state_dict(torch.load("DQN.pt", map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            self.DONE_MEM = torch.load("DONE_MEM.pt")
            # with open("ending_position.pkl", 'rb') as f:
            #     self.ending_position = pickle.load(f)
            # with open("num_in_queue.pkl", 'rb') as f:
            #     self.num_in_queue = pickle.load(f)
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
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    # def act(self, state):
    #     """Epsilon-greedy action"""
    #     if random.random() < self.exploration_rate:
    #         # Select a random action
    #         # TODO:
    #         action = torch.tensor([random.uniform(-1, 1), random.uniform(-1, 1)])
    #     else:
    #         # Select the action with highest Q-value
    #         q_values = self.dqn(state.to(self.device))
    #         action = q_values.squeeze().cpu().detach()
    #     return action

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
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)


def run(training_mode, pretrained, num_episodes=1000, exploration_max=1):
    # env = gym.make('Breakout-v0') # can change the environmeent accordingly
    # env = create_env(env)  # Wraps the environment so that frames are grayscale

    game_name = 'running-competition'

    num_agents = 2
    ctrl_agent_index = 0  # controlled agent index

    print(f'Total agent number: {num_agents}')
    print(f'Agent control by the actor: {ctrl_agent_index}')

    load_model = False
    run_dir, log_dir = make_logpath(game_name, "DuelingDQN")
    if game_name == 'running-competition':
        map_id = random.randint(1, 4)
        # map_id = 3
        Gamemap = create_scenario(game_name)
        env = Running_competition(meta_map=Gamemap, map_id=map_id, vis=200, vis_clear=5, agent1_color='light red',
                                  agent2_color='blue')
        env.max_step = 400
    elif game_name == 'table-hockey':
        Gamemap = create_scenario(game_name)
        env = table_hockey(Gamemap)
        env.max_step = 400
    elif game_name == 'football':
        Gamemap = create_scenario(game_name)
        env = football(Gamemap)
        env.max_step = 400
    elif game_name == 'wrestling':
        Gamemap = create_scenario(game_name)
        env = wrestling(Gamemap)
        env.max_step = 400
    else:
        raise NotImplementedError

    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
    observation_space = observation_space.shape

    # observation_space = shape=(40, 40, 1)
    action_space = 36
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.2,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     pretrained=pretrained)

    opponent_agent = random_agent()  # we use random opponent agent here

    # Restart the enviroment for each episode
    num_episodes = num_episodes
    env.reset()

    total_rewards = []
    # if training_mode and pretrained:
    #     with open("total_rewards.pkl", 'rb') as f:
    #         total_rewards = pickle.load(f)

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        # state = torch.Tensor([state])
        total_reward = 0
        steps = 0

        if isinstance(state[ctrl_agent_index], type({})):
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'], env.agent_list[
                ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - ctrl_agent_index]['agent_obs'], env.agent_list[
                1 - ctrl_agent_index].energy
        else:
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1 - ctrl_agent_index], env.agent_list[1 - ctrl_agent_index].energy

        obs_ctrl_agent = torch.Tensor([obs_ctrl_agent])  # (40, 40) -> torch.Size([1, 40, 40])

        while True:
            action = agent.act(obs_ctrl_agent)
            action_ctrl = action
            action_opponent = [0, 0]
            # action_ctrl_agent= action.data.tolist()
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]
            steps += 1

            # state_next, reward, terminal, info = env.step(int(action[0]))
            next_state, reward, terminal, info = env.step(action)

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

            next_obs_ctrl_agent = torch.Tensor([next_obs_ctrl_agent])  # (40, 40) -> torch.Size([1, 40, 40])

            total_reward += reward[ctrl_agent_index] if terminal else -1
            # next_state = torch.Tensor([next_state])
            # reward_ctrl = torch.tensor([reward]).unsqueeze(0)
            reward_ctrl = torch.Tensor([reward[ctrl_agent_index]])

            # TODO: maybe not good
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            # obs_ctrl_agent_to_remamber = obs_ctrl_agent.permute(1, 2, 0)

            if training_mode:
                # agent.remember(obs_ctrl_agent.permute(1, 2, 0), action_ctrl, reward_ctrl, next_obs_ctrl_agent.permute(1, 2, 0), terminal)
                agent.remember(obs_ctrl_agent, action_ctrl, reward_ctrl, next_obs_ctrl_agent, terminal)
                agent.experience_replay()

            state = next_state

            obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
            obs_ctrl_agent, energy_ctrl_agent = next_obs_ctrl_agent, next_energy_ctrl_agent

            if terminal:
                break

        total_rewards.append(total_reward)

        if ep_num != 0 and ep_num % 100 == 0:
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1],
                                                                     np.mean(total_rewards)))
        num_episodes += 1

    print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))

    # # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    # if training_mode:
    #     with open("ending_position.pkl", "wb") as f:
    #         pickle.dump(agent.ending_position, f)
    #     with open("num_in_queue.pkl", "wb") as f:
    #         pickle.dump(agent.num_in_queue, f)
    #     with open("total_rewards.pkl", "wb") as f:
    #         pickle.dump(total_rewards, f)

    #     torch.save(agent.dqn.state_dict(), "DQN.pt")
    #     torch.save(agent.STATE_MEM,  "STATE_MEM.pt")
    #     torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
    #     torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
    #     torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
    #     torch.save(agent.DONE_MEM,   "DONE_MEM.pt")

    env.close()


# for training
if __name__ == "__main__":
    run(training_mode=True, pretrained=False)

# # for testing
# run(training_mode=False, pretrained=True, num_episodes=1, exploration_max=0.05)