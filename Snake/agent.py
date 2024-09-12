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


# TODO: Add model loading
# Define Agent with Experience Replay Buffer
class SnakeAgent:
    def __init__(self, width, height, action_dim, model_type='linear', checkpoint=None, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=100000, min_memory_size=10000):
        self.width = width
        self.height = height
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.min_memory_size = min_memory_size
        self.episode_at_last_checkpoint = None
        # Check if MPS is available and set device accordingly
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.load_model(model_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_checkpoint(checkpoint)

    # TODO: Should find a way to get the episode number back from the model save
    def load_checkpoint(self, checkpoint):
        # Get model parameters from checkpoint if any
        if checkpoint is not None:
            model_dic = torch.load(checkpoint)
            self.model.load_state_dict(model_dic['model_state_dict'])
            self.optimizer.load_state_dict(model_dic['optimizer_state_dict'])
            self.epsilon = model_dic['epsilon']
            self.memory = model_dic['replay_buffer']
            self.episode_at_last_checkpoint = model_dic['episode']
        else:
            self.episode_at_last_checkpoint = 0

    def load_model(self, model_type):
        # Create model architecture
        if model_type == 'linear':
            # TODO: Get state input dimension / hidden dimension as a variable
            model = SnakeLinearQNet(11, 256, self.action_dim).to(self.device)
        elif model_type == 'cnn':
            model = SnakeCNNQNet(self.width, self.height, self.action_dim).to(self.device)
        else:
            raise NotImplementedError

        return model

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
        # TODO: Y a un truc qui foire avec le min_memory_size
        if len(self.memory) < batch_size or len(self.memory) < self.min_memory_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        minibatch_loss_values = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model.forward(next_state)).item()
            target_f = self.model.forward(state).detach().cpu().numpy()
            target_f[0][action] = target
            target_f = torch.tensor(target_f).to(self.device)
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model.forward(state))
            # TODO: J'ai l'impression qu'il y a un problème ici, est ce que le gradient descent devrait pas se faire
            #  une fois que tout le batch est pris en compte avec genre une moyenne de la loss ?
            loss.backward()
            self.optimizer.step()
            # Store the loss value
            minibatch_loss_values.append(loss.item())
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

        return minibatch_loss_values
