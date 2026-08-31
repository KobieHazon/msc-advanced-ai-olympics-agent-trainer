from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .base_agent_model import BaseAgentModel, device
from ...algo.network import Actor, Critic, CNN_Actor, CNN_Critic


class PPOAgentModel(BaseAgentModel):
    def __init__(self):
        super().__init__()
        self._actor_nn = CNN_Actor(self.STATE_SPACE, self.ACTION_SPACE)
        self._critic_nn = CNN_Critic(self.STATE_SPACE)

        self.training_step = 0

        self.actor_optimizer = optim.Adam(self._actor_nn.parameters(), lr=self.LR)
        self.critic_net_optimizer = optim.Adam(self._critic_nn.parameters(), lr=self.LR)

        self.writer = SummaryWriter("ppo_agent_training_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
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

        states = np.array([t.state.numpy() for t in self.transitions])
        state = torch.tensor(states, dtype=torch.float).to(device)
        # state = torch.tensor(np.array([t.state for t in self.transitions]), dtype=torch.float).to(device)
        action = torch.tensor(np.array([t.action for t in self.transitions]), dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.transitions]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.transitions]), dtype=torch.float).view(-1, 1).to(
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

                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

        del self.transitions[:]  # clear experience

    def get_action(self, state: np.array, train: bool = True) -> Tuple[Tuple[float, float], int]:
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
        return action.item(), action_prob[:, action.item()].item()
