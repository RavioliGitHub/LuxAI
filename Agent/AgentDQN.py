from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import statistics
from IO.ConfigReader import read_yaml_file


# Memory for experience replay of the agent
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# Class used to plot training status
class Plotter:
    def __init__(self):
        # Store losses of each optimization step for the current episode
        self.current_episode_losses = []
        # Store current total reward for the episode
        self.current_total_reward = 0
        # Store data associated to each episode
        self.reward_per_episode = []
        self.mean_loss_per_episode = []

    def end_episode(self):
        # Store episode reward
        self.reward_per_episode.append(self.current_total_reward)
        # Store episode mean loss
        episode_mean_loss = statistics.mean(self.current_episode_losses)
        self.mean_loss_per_episode.append(episode_mean_loss)
        # Print training status
        print(f"Episode: {len(self.reward_per_episode) + 1}, "
              f"Total Reward: {self.current_total_reward}, "
              f"Episode mean loss: {episode_mean_loss}")
        # Reset episode data
        self.current_total_reward = 0
        self.current_episode_losses = []


# Agent trained with Deep Q Learning (DQN) algorithm
class AgentDQN:
    def __init__(self, game, trainer, policy_network, target_network, optimizer, config='../Agent/AgentDQN_config.yaml'):
        """

        :param config:
        :param game: Game object which must have: reset() function which resets the game
        play_step() function which plays a step of the game based on an input action and returns the corresponding
        reward and if the episode has finished or not
        :param trainer: Trainer object which must have the get_game_state() function, and an "action_dim" attribute
        in its config
        :param policy_network:
        :param target_network:
        """
        # Load DQN agent config
        self.config = read_yaml_file(config)
        # Store game object
        self.game = game
        # Store trainer object
        self.trainer = trainer
        # Store both models and optimizer
        self.policy_network = policy_network
        self.target_network = target_network
        self.optimizer = optimizer
        # Init agent memory
        self.memory = ReplayMemory(maxlen=self.config['replay_memory_size'])
        # Create Plotter object to store training data
        self.plotter = Plotter()

    # Returns an integer value corresponding to one of the possible actions
    def act(self, state):
        # Random action
        if np.random.rand() <= self.config['epsilon']:
            return np.random.choice(self.trainer.config['action_dim'])
        # If the policy network determines the next action
        q_values = self.policy_network.forward(state)
        return torch.argmax(q_values).item()  # Returns the index of the max value as an integer

    def optimize(self, minibatch):
        # Store the sets of Q values for the current minibatch
        policy_q_list = []
        target_q_list = []

        for state, action, reward, next_state, done in minibatch:
            # Compute target Q value based on DQN algorithm
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.config['gamma'] * torch.max(self.target_network.forward(next_state)).item()

            # Get the target set of Q values
            target_q = self.target_network.forward(state)
            target_q[0, action] = target
            target_q_list.append(target_q)
            # Get the policy set of Q values
            policy_q = self.policy_network.forward(state)
            policy_q_list.append(policy_q)

        # Compute loss for the whole minibatch
        loss = nn.MSELoss()(torch.stack(policy_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store the loss value for plotting
        self.plotter.current_episode_losses.append(loss.item())

    def replay(self):
        # Sample minibatch from the memory and optimize the policy network
        minibatch = self.memory.sample(self.config['mini_batch_size'])
        self.optimize(minibatch)

        # Decay epsilon
        if self.config['epsilon'] > 0.01:
            self.config['epsilon'] *= self.config['epsilon_decay']

    # Pre-fill memory with random actions, so the model can be optimized at episode 1
    def fill_initial_memory(self):
        while (len(self.memory) < self.config['mini_batch_size']
                or len(self.memory) < self.config['min_replay_memory_size']):
            self.game.reset()
            # Flag to check if episode has ended
            done = False
            # Get current game state
            state = self.trainer.get_game_state()
            # Iterate until end of episode
            while not done:
                # Select action based on epsilon-greedy
                action = self.act(state)
                # Make the action in-game
                reward, done = self.game.play_step(action)
                # Get game state after the action was taken
                if not done:
                    next_state = self.trainer.get_game_state()
                else:
                    next_state = None
                # Save experience into memory
                self.memory.append((state, action, reward, next_state, done))
                state = next_state

    def train_agent(self):
        # Used to keep track of when to sync policy and target network
        step_count = 0
        # Pre-fill memory
        self.fill_initial_memory()
        # Start training loop
        for episode in range(self.config['nb_episodes']):
            # Start new episode
            self.game.reset()
            # Flag to check if episode has ended
            done = False
            # Get current game state
            state = self.trainer.get_game_state()
            # Iterate until end of episode
            while not done:
                # Select action based on epsilon-greedy
                action = self.act(state)
                # Make the action in-game
                reward, done = self.game.play_step(action)
                # Get game state after the action was taken
                if not done:
                    next_state = self.trainer.get_game_state()
                else:
                    next_state = None
                # Save experience into memory
                self.memory.append((state, action, reward, next_state, done))
                # Move to the next state
                step_count += 1
                state = next_state
                self.plotter.current_total_reward += reward
                # Optimize the agent
                self.replay()
                # Copy policy network to target network after a certain number of steps
                if step_count > self.config['network_sync_rate']:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    step_count = 0

            # Display training status at the end of episode
            self.plotter.end_episode()
