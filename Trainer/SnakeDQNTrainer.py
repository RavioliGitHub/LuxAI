from Games.SnakeGame import SillySnakeGameAi
from Models.SnakeModels import SnakeLinearQNet, SnakeCNNQNet
import torch.optim as optim
from Agent.AgentDQN import AgentDQN
from IO.ConfigReader import read_yaml_file


class SnakeDQNTrainer:
    def __init__(self, config='SnakeDQNTrainer_config.yaml', model_type='linear', device='cpu', checkpoint=None):
        self.config = read_yaml_file(config)
        # Init snake game
        self.game = SillySnakeGameAi(width=self.config['width'],
                                     height=self.config['height'],
                                     playerName="Lucas")

        # Init models
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.policy_network = None
        self.target_network = None
        self.optimizer = None
        self.load_models()

        # Init agent
        self.agent = AgentDQN(game=self.game,
                              trainer=self,
                              policy_network=self.policy_network,
                              target_network=self.target_network,
                              optimizer=self.optimizer)

    def init_models(self):
        # Create new random model
        if self.model_type == 'linear':
            return SnakeLinearQNet(config='../Models/SnakeModels_config.yaml',
                                   num_actions=self.config['action_dim']).to(self.device)
        elif self.model_type == 'cnn':
            return SnakeCNNQNet(self.config['width'], self.config['height'], self.config['action_dim']).to(self.device)
        else:
            raise NotImplementedError

    def load_models(self):
        if self.checkpoint is None:  # Create new random networks
            # Create policy and target network
            self.policy_network = self.init_models()
            self.target_network = self.init_models()
            # Define optimizer
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config['lr'])
            # Copy policy network parameters and apply them to target network
            self.target_network.load_state_dict(self.policy_network.state_dict())
        else:  # Load checkpoint for the models
            pass

    # This function will be used by the agent for the training
    # It must return current game state
    def get_game_state(self):
        if self.model_type == 'linear':
            state = self.game.get_simplified_game_state()
        elif self.model_type == 'cnn':
            state = self.game.get_game_grid()
            state = state.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError

        return state.to(self.device)

    # Train the Snake AI
    def train_snake(self):
        self.agent.train_agent()


if __name__ == "__main__":
    trainer = SnakeDQNTrainer(model_type='linear')
    trainer.train_snake()
