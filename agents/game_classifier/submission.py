import os
import sys
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import gym
import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}  # dicretise action space

observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
observation_space = observation_space.shape

# observation_space = shape=(40, 40, 1)
action_space = len(actions_map)



sys.path.pop(-1)  # just for safety
import random

import numpy as np
from torch import nn, optim


# PPO Agent Model
@dataclass
class Transition:
    state: np.array
    action: int
    a_log_prob: int
    reward: int
    next_state: np.array
    is_done: bool


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
    def get_action(self, state: np.array):
        pass

    def save(self, save_file_path: str):
        model_actor_path = os.path.join(save_file_path, "running_actor.pth")
        model_critic_path = os.path.join(save_file_path, "critic.pth")

        torch.save(self.actor_nn.state_dict(), model_actor_path)
        torch.save(self.critic_nn.state_dict(), model_critic_path)

    def load(self, actor_file_path: str):
        if os.path.exists(actor_file_path):
            actor = torch.load(actor_file_path, map_location=device)
            self.actor_nn.load_state_dict(actor)
        else:
            raise ValueError("Tried to load models from files that don't exist")


class CNN_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Actor, self).__init__()

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

        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_net(x)

        # Calculate the softmax probabilities
        return F.softmax(value, dim=-1)


class CNN_Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64):
        super(CNN_Critic, self).__init__()

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

        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.value_net(x)

        # Calculate the softmax probabilities
        return value


class PPOAgentModel(BaseAgentModel):
    def __init__(self):
        super().__init__()
        self._actor_nn = CNN_Actor(self.STATE_SPACE, self.ACTION_SPACE)
        self._critic_nn = CNN_Critic(self.STATE_SPACE)

        self.training_step = 0

        self.actor_optimizer = optim.Adam(self._actor_nn.parameters(), lr=self.LR)
        self.critic_net_optimizer = optim.Adam(self._critic_nn.parameters(), lr=self.LR)

        self.IO = True

    @property
    def actor_nn(self) -> nn.Module:
        return self._actor_nn

    @property
    def critic_nn(self) -> nn.Module:
        return self._critic_nn

    def update_result(self, is_win: bool):
        if not is_win:
            del self.transitions[:]
            return

        state = torch.tensor(np.array([t.state for t in self.transitions]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.transitions]), dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.transitions]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.transitions]), dtype=torch.float).view(
            -1, 1).to(
            device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.GAMMA * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        # print("The agent is updateing....")
        for i in range(self.PPO_UPDATE_TIME):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.transitions))), self.BATCH_SIZE, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，is_train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self._critic_nn(state[index].squeeze(1))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self._actor_nn(state[index].squeeze(1)).gather(1, action[index])  # new policy

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.CLIP_PARAM, 1 + self.CLIP_PARAM) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self._actor_nn.parameters(), self.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # update critic network
                value_loss = torch.nn.functional.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self._critic_nn.parameters(), self.MAX_GRAD_NORM)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.transitions[:]  # clear experience

    def get_action(self, state: np.array, train: bool = True):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = state.to(device)
        with torch.no_grad():
            action_prob = self._actor_nn(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action


# Random Agent Model
class RandomAgentModel():
    def __init__(self):
        self.min_action_index = min(actions_map.keys())
        self.max_action_index = max(actions_map.keys())

    def get_action(self, state: np.array):
        return torch.Tensor([random.randrange(self.min_action_index, self.max_action_index + 1)])


# Game Classifier Model:
class GameClassifierNet(nn.Module):
    def __init__(self):
        super(GameClassifierNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)  # softmax layer
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # if you have a GPU, otherwise it will use CPU
game_classifier_model = GameClassifierNet().to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(game_classifier_model.parameters(), lr=0.01)

game_classifier_model_file_path = os.path.dirname(
    os.path.abspath(__file__)) + f"{os.sep}model-game-classifier-loss023-acc-0905.pth"
game_classifier_model.load_state_dict(torch.load(game_classifier_model_file_path))
game_classifier_model.eval()

GAME_CLASSIFIER_ID_TO_AGENT = {
    0: PPOAgentModel(),  # Football, TODO: change
    1: RandomAgentModel(),  # table-hockey
    2: RandomAgentModel(),  # curling
    3: RandomAgentModel(),  # wrestling
    4: PPOAgentModel(),  # running
}
GAME_CLASSIFIER_ID_TO_AGENT[0].load(os.path.dirname(os.path.abspath(__file__)) + f"{os.sep}football_actor.pth")
GAME_CLASSIFIER_ID_TO_AGENT[4].load(os.path.dirname(os.path.abspath(__file__)) + f"{os.sep}running_actor.pth")

# summary(model, (1, 40, 40))
def my_controller(obs_list, action_space_list, obs_space_list):
    state = obs_list['obs']['agent_obs']
    input_data = torch.from_numpy(state).unsqueeze(0).unsqueeze(1).float().to(device)
    if state.shape != (40, 40):
        agent = RandomAgentModel()  # The game is billiard
    else:
        with torch.no_grad():
            output = game_classifier_model(input_data)
            predicted_game = torch.argmax(output, dim=1).cpu().numpy()[0]
            agent = GAME_CLASSIFIER_ID_TO_AGENT[predicted_game]
    with torch.no_grad():
        input_data = input_data.squeeze(0)
        action = agent.get_action(input_data)
    action_index = int(action.data.tolist()[0])
    action_data = actions_map[action_index]
    action_data = [[action_data[0]], [action_data[1]]]
    return action_data
