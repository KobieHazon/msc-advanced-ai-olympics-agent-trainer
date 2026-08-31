import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from olympics_engine.generator import create_scenario
from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition
from train.agents.duelingdqn_agent import DuelingDQNAgent
from train.agents.random_agent import random_agent

# Game Classifier network
class GameClassifier(nn.Module):
    def __init__(self, num_of_games):
        super(GameClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_of_games),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_game_classifier(game_classifier, optimizer, criterion, games, agent, ctrl_agent_index, num_episodes=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_classifier.to(device)
    agent.to(device)

    for epoch in range(num_episodes):
        correct_predictions = 0
        total_predictions = 0

        for game_id, env in enumerate(games):
            state = env.reset()
            done = False

            if isinstance(state[ctrl_agent_index], dict):
                obs_ctrl_agent = state[ctrl_agent_index]['agent_obs']
            else:
                obs_ctrl_agent = state[ctrl_agent_index]

            obs_ctrl_agent = np.stack([obs_ctrl_agent, obs_ctrl_agent], axis=0)  # Duplicate the observation
            obs_ctrl_agent = torch.from_numpy(obs_ctrl_agent).float().unsqueeze(0).to(device)  # Add batch dimension and move to the device

            while not done:
                action = agent.act(obs_ctrl_agent)

                action_digit = action.data.tolist()
                action_ctrl = actions_map[action_digit]

                action_combined = [action_ctrl, [0, 0]]  # Agent's action combined with opponent's action

                next_state, _, done, _ = env.step(action_combined)

                if isinstance(next_state[ctrl_agent_index], dict):
                    next_obs_ctrl_agent = next_state[ctrl_agent_index]['agent_obs']
                else:
                    next_obs_ctrl_agent = next_state[ctrl_agent_index]

                next_obs_ctrl_agent = np.expand_dims(next_obs_ctrl_agent, axis=0)  # Add batch dimension
                next_obs_ctrl_agent = torch.from_numpy(next_obs_ctrl_agent).to(device)  # Move to the device

                optimizer.zero_grad()

                outputs = game_classifier(obs_ctrl_agent)
                loss = criterion(outputs, torch.tensor([game_id]).to(device))
                loss.backward()
                optimizer.step()

                obs_ctrl_agent = next_obs_ctrl_agent

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += predicted.size(0)
                correct_predictions += (predicted == game_id).sum().item()

        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}")

    print('Finished Training')
    torch.save(game_classifier.state_dict(), 'game_classifier.pth')


# Initialize games
game_map1 = create_scenario('running-competition')
game_map2 = create_scenario('table-hockey')
game_map3 = create_scenario('football')
game_map4 = create_scenario('wrestling')
games = [
    Running_competition(meta_map=game_map1, map_id=1, vis=200, vis_clear=5, agent1_color='light red', agent2_color='blue'),
    table_hockey(game_map2),
    football(game_map3),
    wrestling(game_map4)
]

# Initialize agent
observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 40, 40), dtype=np.uint8)
observation_space = observation_space.shape

# discrete action space
actions_map = {
    0: [100, 0],  # N
    1: [100, 30],  # NE
    2: [100, -30],  # NW
    3: [-100, 0],  # S
    4: [-100, 30],  # SW
    5: [-100, -30],  # SE
}
action_space = len(actions_map)

# Initialize agent
agent = DuelingDQNAgent(state_space=observation_space,
                        action_space=action_space,
                        max_memory_size=50000,
                        batch_size=64,
                        gamma=0.95,
                        lr=0.0005,
                        dropout=0.2,
                        exploration_max=1.0,
                        exploration_min=0.02,
                        exploration_decay=0.995,
                        pretrained=False)

# Initialize classifier
num_games = len(games)
game_classifier = GameClassifier(num_games)

# Initialize optimizer and loss function
optimizer = optim.Adam(game_classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train game classifier
ctrl_agent_index = 0  # Index of the controlled agent
train_game_classifier(game_classifier, optimizer, criterion, games, agent, ctrl_agent_index)
