import torch.nn as nn
import torch.nn.functional as f


class DQN_Net(nn.Module):
    def __init__(self, num_actions, dropout):
        """
        Initialize
        Arguments:
            num_actions: amount of action-value to output, one-to-one correspondence to action in game.
            Dropout: dropout rate for dropout layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.fc4 = nn.Linear(64, 512)  # Update the input size of fc4
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(f.relu(self.fc4(x)))  # Apply dropout after ReLU
        return f.softmax(self.fc5(x), dim=1)
