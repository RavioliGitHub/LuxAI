from Snake.SnakeGame import SillySnakeGameAi, BLOCK_SIZE
from Snake.agent import SnakeAgent
import torch


# TODO: Transformer ça en classe "trainer" pcq dégueulasse comme fonction
def playGame(playerName = "Stefan", model='linear'):
    # Create game
    game = SillySnakeGameAi(playerName=playerName)
    # Create agent
    agent = SnakeAgent(width=int(game.w//BLOCK_SIZE), height=int(game.h//BLOCK_SIZE), action_dim=3)

    # Check if MPS is available and set device accordingly
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Train the agent with Experience Replay Buffer
    batch_size = 100
    num_episodes = 100000
    for episode in range(num_episodes):
        # Start new episode
        game.reset()
        total_reward = 0
        done = False
        # Get current state
        if model == 'linear':
            state = game.get_simplified_game_state()
        elif model == 'cnn':
            state = game.get_game_grid()
            state = state.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError
        state = state.to(device)

        while not done:
            # Get action based on current state
            # TODO: Il y a un peu de confision entre decision / action --> il faudra corriger,
            #  en théorie l'action devrait être l'index, faudrait pas d'intermédiaire (et plutôt adapter le code de
            #  pygame qui de base demande une liste et pas un index)
            decision = agent.act(state)
            # Make the action
            action = [0, 0, 0]
            action[decision] = 1
            reward, done, _ = game.playStep(action)
            if not done and model == 'linear':
                next_state = game.get_simplified_game_state()
                next_state = next_state.to(device)
            elif not done and model == 'cnn':
                next_state = game.get_game_grid()
                next_state = next_state.unsqueeze(0).unsqueeze(0)
                next_state = next_state.to(device)
            else:
                next_state = None
            # Store in memory
            agent.remember(state, decision, reward, next_state, done)
            # Continue
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# include a main function, to be able to run the script, for notebooks we can comment this out
if __name__ == "__main__":
    playGame()