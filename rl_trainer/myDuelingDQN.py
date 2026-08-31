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



# main = [0, 0.2, 0.4, 0.6, 0.8, 1] # all the way to 1
# sides = [-1, -0.9, -0.8, -0.7, -0.6 ,-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# combinations = []
# for x in main:
#     for y in sides:
#         combinations.append([x, y])


# def discrete_to_continuous(value):
#     return combinations[value]


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RENDER = False

class DuelingDQNAgent:

    def __init__(self, state_size, action_size, seed, parameters):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.parameters = parameters
        
        self.qnetwork_stable = DuelingQNetwork(state_size, action_size, seed)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed)

        # self.qnetwork_stable = DuelingDQN(action_size, device).to(device)
        # self.qnetwork_target = DuelingDQN(action_size, device).to(device)

        # self.qnetwork_stable = ConvDuelingDQN(state_size, action_size)
        # self.qnetwork_target = ConvDuelingDQN(state_size, action_size)

        
        # self.qnetwork_stable = QNetwork().to(device)
        # self.qnetwork_target = QNetwork().to(device)

        self.optimizer = optim.Adam(self.qnetwork_stable.parameters(), lr=self.parameters.LR)

        # replay memory
        self.memory = deque(maxlen=self.parameters.BUFFER_SIZE)  
        self.batch_size = self.parameters.BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # timestep
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        self.add_to_memory(state, action, reward, next_state, done)
        
        if self.t_step + 1 == self.parameters.UPDATE_EVERY:
            self.t_step = 0
        
        if self.t_step == 0:
            
            if len(self.memory) > self.parameters.BATCH_SIZE:
                self.learn(self.sample_from_memory(), self.parameters.GAMMA)

    
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_stable.eval()
        
        with torch.no_grad() as nograd:
            action_values = self.qnetwork_stable(state)

        self.qnetwork_stable.train()

        # exploration-explotation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
              
    def learn(self, raw_memory_experience, gamma):
        
        loss = self.compute_loss(raw_memory_experience, gamma)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # pdate target network
        self.soft_update(self.qnetwork_stable, self.qnetwork_target, self.parameters.TAU)  

    def compute_loss(self, experiences, gamma):
        states, actions, rewards, next_states, done_array = experiences

        Q_argmax = self.qnetwork_stable(next_states).detach()

        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_argmax.max(1)[1].unsqueeze(1))
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done_array))


        Q_expected = self.qnetwork_stable(states).gather(1, actions)

        # loss
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def soft_update(self, local_network, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))
    
    def sample_from_memory(self):
        # Randomly sample a batch of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)
        clean_exp = []
        for e in experiences:
            if e is not None:
                clean_exp.append(e)

        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences])).float()
        states = states.to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        actions = actions.to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        rewards = rewards.to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state.cpu() for e in experiences])).float()
        next_states = next_states.to(device)
        done_array = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        done_array = done_array.to(device)

        return (states, actions, rewards, next_states, done_array)
    
    def save(self, save_path, episode, score):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_qnetwork_stable_path = os.path.join(base_path, "qnetwork_stable_" + str(episode) + ".pth")
        torch.save(self.qnetwork_stable.state_dict(), model_qnetwork_stable_path)
        model_qnetwork_target_path = os.path.join(base_path, "qnetwork_target_" + str(episode) + ".pth")
        torch.save(self.qnetwork_target.state_dict(), model_qnetwork_target_path)

        model_qnetwork_stable_path_score = os.path.join(base_path, "qnetwork_stable_" + str(episode) + "score_" +str(score) + ".pth")
        torch.save(self.qnetwork_stable.state_dict(), model_qnetwork_stable_path_score)
        model_qnetwork_target_path_score = os.path.join(base_path, "qnetwork_target_" + str(episode) + "score_" +str(score) + ".pth")
        torch.save(self.qnetwork_target.state_dict(), model_qnetwork_target_path_score)

    def load(self, run_dir,game_name, algo, episode):

        base_dir = Path(__file__).resolve().parent
        model_dir = base_dir / Path('models') / game_name / algo / run_dir / "trained_model"

        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        # base_path = os.path.dirname(os.path.dirname(__file__))
        # print("base_path: ", base_path)
        # algo_path = os.path.join(base_path, 'models/ppo')
        # run_path = os.path.join(algo_path, run_dir)
        # run_path = os.path.join(run_path, 'trained_model')
        # model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        # model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")

        model_qnetwork_stable_path = os.path.join(model_dir, "qnetwork_stable_" + str(episode) + ".pth")
        model_qnetwork_target_path = os.path.join(model_dir, "qnetwork_target_" + str(episode) + ".pth")
        
        print(f'qnetwork_stable path: {model_qnetwork_stable_path}')
        print(f'target_path path: {model_qnetwork_target_path}')

        if os.path.exists(model_qnetwork_stable_path) and os.path.exists(model_qnetwork_target_path):
            qnetwork_stable = torch.load(model_qnetwork_stable_path, map_location=device)
            qnetwork_target = torch.load(model_qnetwork_target_path, map_location=device)
            self.qnetwork_stable.load_state_dict(qnetwork_stable)
            self.qnetwork_target.load_state_dict(qnetwork_target)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')


def train(parameters):

    number_of_episodes = parameters.n_episodes
    max_t = parameters.max_t
    eps_start = parameters.eps_start
    eps_end = parameters.eps_end
    eps_decay = parameters.eps_decay

    # env = gym.make('LunarLanderContinuous-v2')
    # env.seed(0)

    num_agents = 2
    ctrl_agent_index = 0        #controlled agent index

    load_model = False
    run_dir, log_dir = make_logpath(parameters.game_name, "DuelingDQN")
    if parameters.game_name == 'running-competition':
        map_id = random.randint(1,4)
        # map_id = 3
        Gamemap = create_scenario(parameters.game_name)
        env = Running_competition(meta_map=Gamemap,map_id=map_id, vis = 200, vis_clear=5, agent1_color = 'light red',
                                   agent2_color = 'blue')
        env.max_step = 400
    elif parameters.game_name == 'table-hockey':
        Gamemap = create_scenario(parameters.game_name)
        env = table_hockey(Gamemap)
        env.max_step = 400
    elif parameters.game_name == 'football':
        Gamemap = create_scenario(parameters.game_name)
        env = football(Gamemap)
        env.max_step = 400
    elif parameters.game_name == 'wrestling':
        Gamemap = create_scenario(parameters.game_name)
        env = wrestling(Gamemap)
        env.max_step = 400
    else:
        raise NotImplementedError

    

    shape_of_action = 36
    
    agent = DuelingDQNAgent(state_size=1600, action_size=shape_of_action, seed=0, parameters=parameters)
    
    if load_model:
        last_prev_episode_iter = 2000
        agent.load("run2",parameters.game_name, "DuelingDQN",last_prev_episode_iter)
    else:
        last_prev_episode_iter = 0
    
    opponent_agent = random_agent()     #we use random opponent agent here

    scores = []
    solve_iterations_num = 100
    score_memory = deque(maxlen=solve_iterations_num)
    epsilon = eps_start          
    for episode_iter in range(last_prev_episode_iter, number_of_episodes):
        
        state, score = reset_episode(env)

        if RENDER:
            env.render()
        if isinstance(state[ctrl_agent_index], type({})):
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
        else:
            obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index].flatten(), env.agent_list[ctrl_agent_index].energy
            obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy
        
        # for t in range(max_t):
        while True:
            action_ctrl_raw = agent.act(obs_ctrl_agent, epsilon)

            action_opponent = [0,0]  #here we assume the opponent is not moving in the demo

            # continuousAction = discrete_to_continuous(action)
            action_ctrl = actions_map[action_ctrl_raw]

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _ = env.step(action)

            

            if isinstance(next_state[ctrl_agent_index], type({})):
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
            else:
                next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
                next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy


            reward_ctrl_agent = reward[ctrl_agent_index] if done else -1
            agent.step(obs_ctrl_agent, action_ctrl_raw, reward_ctrl_agent,  np.array(next_obs_ctrl_agent).flatten(), done)


            obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
            obs_ctrl_agent, energy_ctrl_agent = np.array(next_obs_ctrl_agent).flatten(), next_energy_ctrl_agent


            state = next_state
            # # adding noise
            # noise = np.random.normal(0,0.05,2)
            # state[0] = state[0] + noise[0]
            # state[1] = state[1] + noise[1]
            score += reward[ctrl_agent_index] if done else -1

            if RENDER:
                env.render()

            if done:
                break 
        score_memory.append(score)
        scores.append(score)
        # epsilon decay
        epsilon = max(eps_end, eps_decay*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_iter, np.mean(score_memory)), end="")
        if episode_iter % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_iter, np.mean(score_memory)))

        # if np.mean(score_memory)>=1.0:
        if episode_iter % 100 == 0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_iter-solve_iterations_num, np.mean(score_memory)))
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_Dueling_DDQN.pth')
            agent.save(run_dir, episode_iter, score)

            # break

    return scores

def reset_episode(env):
    state = env.reset()
    score = 0
    return state, score


def make_logpath(game_name, algo):
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / Path('models') / game_name / algo
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir
    return run_dir, log_dir



# class DuelingQNetwork(nn.Module):

#     def __init__(self, input_dim, output_dim, seed):
#         super(DuelingQNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.seed = torch.manual_seed(seed)

#         self.feauture_layer = nn.Sequential(
#             nn.Linear(self.input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU()
#         ).to(device)
        
#         self.value_stream = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         ).to(device)

#         self.advantage_stream = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, self.output_dim)
#         ).to(device)

#     def forward(self, state):
#         features = self.feauture_layer(state)
#         values = self.value_stream(features)
#         advantages = self.advantage_stream(features)
#         qvals = values + (advantages - advantages.mean(1).unsqueeze(1).expand(state.size(0), self.output_dim))
        
#         return qvals


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4,2),
            nn.Flatten()
        )

    def forward(self, view_state):
        # [batch, 128]
        x = self.net(view_state)
        return x

class DuelingQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, seed):
        super(DuelingQNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = torch.manual_seed(seed)

        self.encoder = CNN_encoder().to(device)

        self.feauture_layer = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        ).to(device)
        
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        ).to(device)

    def forward(self, state):
        state = self.reshapeTensor(state)
        state = self.encoder(state)
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(1).unsqueeze(1).expand(state.size(0), self.output_dim))
        
        return qvals

    # def reshapeTensor2(self, tensor):
    #     # Assuming the original tensor is called input_tensor
    #     batch_size = 1
    #     num_channels = 1
    #     height = 40
    #     width = 40

    #     # Reshape the input tensor
    #     tensor = tensor.reshape(batch_size, num_channels, height, width)
    #     return tensor
    
    def reshapeTensor(self, tensor, new_height=40, new_width=40):

        # get the size of the dimensions of the tensor
        sizes = tensor.size()

        # create a batch of images, each with 1600 elements
        batch_size = sizes[0]

        # reshape each image in the batch to have 1 channel, 40x40 pixels
        new_channels = 1
        # new_height, new_width = 40, 40

        # create a new tensor with the desired shape
        reshaped_images = torch.empty(batch_size, new_channels, new_height, new_width).to(device)

        # iterate over the batch and reshape each image
        for i in range(batch_size):
            image = tensor[i].view(new_channels, new_height, new_width).to(device)
            reshaped_images[i] = image

        return reshaped_images



class DuelingDQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DuelingDQN, self).__init__()
        self.__conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc_v = nn.Linear(64*7*7, 512)
        self.__v = nn.Linear(512, 1)
        self.__fc_advt = nn.Linear(64*7*7, 512)
        self.__advt = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        x = self.reshapeTensor(x)
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        v = F.relu(self.__fc_v(x.view(x.size(0), -1)))
        v = self.__v(v)
        advt = F.relu(self.__fc_advt(x.view(x.size(0), -1)))
        advt = self.__advt(advt)
        return v + (advt - advt.mean(dim=1, keepdim=True))

    def reshapeTensor(self, tensor, new_height=40, new_width=40):

        # get the size of the dimensions of the tensor
        sizes = tensor.size()

        # create a batch of images, each with 1600 elements
        batch_size = sizes[0]

        # reshape each image in the batch to have 1 channel, 40x40 pixels
        new_channels = 1
        # new_height, new_width = 40, 40

        # create a new tensor with the desired shape
        reshaped_images = torch.empty(batch_size, new_channels, new_height, new_width).to(device)

        # iterate over the batch and reshape each image
        for i in range(batch_size):
            image = tensor[i].view(new_channels, new_height, new_width).to(device)
            reshaped_images[i] = image

        return reshaped_images

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


# class ConvDuelingDQN(nn.Module):

#     def __init__(self, input_dim, output_dim):
#         super(ConvDuelingDQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc_input_dim = self.feature_size()
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self.value_stream = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#         self.advantage_stream = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.output_dim)
#         )

#     def forward(self, state):
#         features = self.conv(state)
#         features = features.view(features.size(0), -1)
#         values = self.value_stream(features)
#         advantages = self.advantage_stream(features)
#         qvals = values + (advantages - advantages.mean())
        
#         return qvals

#     def feature_size(self):
#         return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


# https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/visual_doom.py
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # self.resnet = models.resnet18(pretrained=False)
        # # self.resnet = models.resnet50(pretrained=False)
        # self.relu = nn.ReLU()
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 64)
        #
        # self.fc_value = nn.Linear(64, 256)
        # self.fc_adv = nn.Linear(64, 256)
        #
        # self.value = nn.Linear(256, 1)
        # self.adv = nn.Linear(256, 3)

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, 6, stride=2, padding=2)  # 64 * 64 * 3 -> 32 * 32 * 64

        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 32 * 32 * 64 -> 32 * 32 * 64
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 32 * 32 * 64 -> 32 * 32 * 64

        self.conv3 = nn.Conv2d(64, 64, 6, stride=2, padding=2)  # 32 * 32 * 64 -> 16 * 16 * 64

        self.conv4_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  # 16 * 16 * 64 -> 16 * 16 * 64
        self.conv4_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 6, stride=2, padding=2)  # 16 * 16 * 64 -> 8 * 8 * 64

        self.fc = nn.Linear(8 * 8 * 64, 1024)
        self.value = nn.Linear(1024, 1)
        self.adv = nn.Linear(1024, 3)


    def forward(self, x):
        # x = self.resnet(x)
        # value = self.relu(self.fc_value(x))
        # adv = self.relu(self.fc_adv(x))
        #
        # value = self.value(value)
        # adv = self.adv(adv)
        #
        # advAverage = torch.mean(adv, dim=1, keepdim=True)
        # Q = value + adv - advAverage

        x = self.reshapeTensor2(x)

        x = self.relu(self.conv1(x))
        # print(x.shape)

        y = self.relu(self.conv2_1(x))
        y = self.conv2_2(y)
        # print(y.shape)
        x = self.relu(x + y)

        x = self.relu(self.conv3(x))

        y = self.relu(self.conv4_1(x))
        y = self.conv4_2(y)
        x = self.relu(x + y)

        x = self.relu(self.conv5(x))
        # print(x.shape)

        x = self.relu(self.fc(x.view(x.size(0), -1)))


        value = self.value(x)
        adv = self.adv(x)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q
    
    def reshapeTensor2(self, tensor):
        # Assuming the original tensor is called input_tensor
        batch_size = 1
        num_channels = 1
        height = 40
        width = 40

        # Reshape the input tensor
        tensor = tensor.reshape(batch_size, num_channels, height, width)
        return tensor


    # def select_action(self, state):
    #     with torch.no_grad():
    #         Q = self.forward(state)
    #         action_index = torch.argmax(Q, dim=1)
    #     return action_index.item()




def parse_arguments():
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer-size', dest = 'BUFFER_SIZE',type=int, default=int(1e5)) # replay buffer size
    parser.add_argument('--batch-size', dest = 'BATCH_SIZE',type=int, default=int(64)) # batch size
    parser.add_argument('--gamma', dest = 'GAMMA',type=float, default=0.99) # discount factor
    parser.add_argument('--tau', dest = 'TAU',type=float, default=(1e-3)) # for soft update
    parser.add_argument('--lr', dest = 'LR',type=float, default=5e-4) # learning rate 
    parser.add_argument('--update-every', dest = 'UPDATE_EVERY',type=int, default=int(4)) # replay buffer size

    parser.add_argument('--episodes-number', dest = 'n_episodes',type=int, default=int(20000)) # number of episodes
    parser.add_argument('--max-t', dest = 'max_t',type=int, default=int(1000))
    parser.add_argument('--eps-start', dest = 'eps_start',type=float, default=1.0)
    parser.add_argument('--eps-end', dest = 'eps_end',type=float, default=0.01)
    parser.add_argument('--eps-decay', dest = 'eps_decay',type=float, default=0.995)

    parser.add_argument('--game_name', default="running-competition", type=str, help='running-competition/table-hockey/football/wrestling')

    return parser.parse_args()

def main(args):

    parameters = parse_arguments()

    scores = train(parameters)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("graph.png")

if __name__ == '__main__':

    main(sys.argv)