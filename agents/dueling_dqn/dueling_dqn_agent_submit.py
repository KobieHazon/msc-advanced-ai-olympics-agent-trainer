# Dueling DQN Agent - 1 game at a time.
# Reward function (winning oriented):
# 1000 - Steps +  (if won + if lost -)Max(1000 - steps//5 + 10000, 0)
# action space discrete.


import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)

from olympics_engine.agent import *


class DuelingDQN_Net(nn.Module):
    def __init__(self, obs_dim, num_actions, dropout):
        super(DuelingDQN_Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

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
