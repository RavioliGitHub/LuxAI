import torch
import torch.nn as nn
from Snake.model import SnakeLinearQNet, SnakeCNNQNet
import random
import numpy as np
"""
deque is a double-ended queue that allows appending and popping elements from both ends efficiently. 
It's a part of Python's standard library under the collections module.
"""
from collections import deque
import torch.optim as optim


# Define Agent with Experience Replay Buffer
class SnakeAgent:
    def __init__(self, width, height, action_dim, model='linear', lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000):
        self.width = width
        self.height = height
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        # Check if MPS is available and set device accordingly
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if model == 'linear':
            # TODO: Get state input dimension / hidden dimension as a variable
            self.model = SnakeLinearQNet(11, 256, action_dim).to(self.device)
        elif model == 'cnn':
            self.model = SnakeCNNQNet(width, height, action_dim).to(self.device)
        else:
            pass
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        # Random action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        q_values = self.model.forward(state)
        # Returns the index of the max value as an integer
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model.forward(next_state)).item()
            target_f = self.model.forward(state).detach().cpu().numpy()
            target_f[0][action] = target
            target_f = torch.tensor(target_f).to(self.device)
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model.forward(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
