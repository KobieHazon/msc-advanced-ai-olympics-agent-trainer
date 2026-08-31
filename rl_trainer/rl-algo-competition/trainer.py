import numpy as np
import torch
import gym
import random
from tqdm import tqdm

from train.agents.dqn_agent import DQNAgent
from train.agents.duelingdqn_agent import DuelingDQNAgent
from train.agents.random_agent import random_agent
from train.log_path import *

from olympics_engine.generator import create_scenario
from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition


def reward_shaping(reward, steps, terminal, max_reward):
    shaped_reward = max_reward - steps

    if terminal:
        if reward > 0:  # Agent won the game
            shaped_reward += max(max_reward - (steps // 5) + 10000, 0)
        else:  # Agent lost the game
            shaped_reward -= max(max_reward - (steps // 5) + 10000, 0)

    return shaped_reward


def run(training_mode, pretrained, num_episodes=10000):
    game_name = 'running-competition'

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
    run_dir, log_dir = make_logpath(game_name, "NET")
    if game_name == 'running-competition':
        map_id = random.randint(1, 4)
        # map_id = 3
        game_map = create_scenario(game_name)
        env = Running_competition(meta_map=game_map, map_id=map_id, vis=200, vis_clear=5, agent1_color='light red',
                                  agent2_color='blue')
        env.max_step = 400
    elif game_name == 'table-hockey':
        game_map = create_scenario(game_name)
        env = table_hockey(game_map)
        env.max_step = 400
    elif game_name == 'football':
        game_map = create_scenario(game_name)
        env = football(game_map)
        env.max_step = 400
    elif game_name == 'wrestling':
        game_map = create_scenario(game_name)
        env = wrestling(game_map)
        env.max_step = 400
    else:
        raise NotImplementedError

    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
    observation_space = observation_space.shape

    # observation_space = shape=(40, 40, 1)
    action_space = len(actions_map)
    agent = DuelingDQNAgent(state_space=observation_space,
                            action_space=action_space,
                            max_memory_size=50000,  # Increased from 30000 to 50000
                            batch_size=64,  # Increased from 32 to 64
                            gamma=0.95,  # Increased from 0.90 to 0.95
                            lr=0.0005,  # Increased from 0.00025 to 0.0005
                            dropout=0.2,
                            exploration_max=1.0,
                            exploration_min=0.02,
                            exploration_decay=0.995,  # Decreased from 0.99 to 0.995
                            pretrained=pretrained)

    opponent_agent = random_agent()  # we use random opponent agent here

    # Restart the environment for each episode
    num_episodes = num_episodes
    env.reset()

    episodes_won = 0
    total_steps = 0
    scores_squared_sum = 0

    total_rewards = []
    max_reward = 1000

    for current_episode in tqdm(range(num_episodes)):
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

        obs_ctrl_agent = torch.Tensor([obs_ctrl_agent])  # (40, 40) -> torch.Size([1, 40, 40])

        while True:
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

            next_obs_ctrl_agent = torch.Tensor([next_obs_ctrl_agent])  # (40, 40) -> torch.Size([1, 40, 40])

            # Apply new reward function
            total_reward = reward_shaping(reward[ctrl_agent_index], steps, terminal, max_reward)

            if terminal:
                if reward[ctrl_agent_index] > reward[1 - ctrl_agent_index]:  # if agent won
                    episodes_won += 1  # increment episodes won

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

        if current_episode != 0 and current_episode % 100 == 0:
            std_dev = np.sqrt(
                max((scores_squared_sum / current_episode) - (np.mean(total_rewards) ** 2), np.finfo(float).eps))

            print(
                f"\nEp: {current_episode + 1}, Last Score: {total_rewards[-1]:.2f}, Avg Score: {np.mean(total_rewards):.2f}, "
                f"Score Std Dev: {std_dev:.2f}, "
                f"Win Rate: {(episodes_won / current_episode) * 100:.2f}%, "
                f"Avg Step: {total_steps / current_episode:.2f}. ")

            if training_mode:
                torch.save(agent.net.state_dict(), "NET.pt")
                torch.save(agent.STATE_MEM, "STATE_MEM.pt")
                torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
                torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
                torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
                torch.save(agent.DONE_MEM, "DONE_MEM.pt")

    print("Training loop ended :)")


# for training
run(training_mode=True, pretrained=False)
