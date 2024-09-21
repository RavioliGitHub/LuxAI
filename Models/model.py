import torch.nn as nn
import torch.nn.functional as F
from IO.ConfigReader import read_yaml_file


class SnakeLinearQNet(nn.Module):
    def __init__(self, num_actions, config='model_config.yaml'):
        super().__init__()
        # Load config
        self.config = read_yaml_file(config)

        # Fully connected layer
        self.fc1 = nn.Linear(self.config['linear']['input_size'], self.config['linear']['hidden_size'])
        self.fc2 = nn.Linear(self.config['linear']['hidden_size'], num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.fc2(x)

        return action


"""# Define the CNN model for Snake game with non-square input
class SnakeCNNQNet(nn.Module):
    def __init__(self, width, height, num_actions):
        super(SnakeCNNQNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolutions and pooling
        conv_output_width = width // (2 ** 3)  # Divide by 8 due to 3 pooling layers
        conv_output_height = height // (2 ** 3)  # Divide by 8 due to 3 pooling layers
        conv_output_size = conv_output_width * conv_output_height * 64  # 64 channels from the last Conv layer

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, num_actions)  # Output layer for actions

    def forward(self, x):
        # Convolution layers followed by pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Downsample by factor of 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Downsample again
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Downsample again
        #print(x.shape)
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # Fully connected layers
        x = F.relu(self.fc1(x))

        action = self.fc2(x)

        return action"""

# Define the CNN model for Snake game with non-square input
class SnakeCNNQNet(nn.Module):
    def __init__(self, width, height, num_actions):
        super(SnakeCNNQNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the size after convolutions and pooling
        conv_output_size = width * height * 64  # 64 channels from the last Conv layer

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_actions)  # Output layer for actions

    def forward(self, x):
        # Convolution layers followed by pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        #print(x.shape)
        # Flatten the output from conv layers
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        action = self.fc2(x)

        return action
