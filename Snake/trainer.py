from Snake.SnakeGame import SillySnakeGameAi, BLOCK_SIZE
from Snake.agent import SnakeAgent
import torch
import statistics


def save_rl_model(model, optimizer, episode, replay_buffer, epsilon):
    save_path = f'SnakeRL_model.pth'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'replay_buffer': replay_buffer,
                'epsilon': epsilon
                   ,}, save_path)


# TODO: Transformer ça en classe "trainer" pcq dégueulasse comme fonction
def playGame(playerName = "Stefan", model_type='linear', checkpoint=None):
    # Create game
    game = SillySnakeGameAi(playerName=playerName)
    # Create agent
    agent = SnakeAgent(width=int(game.w//BLOCK_SIZE),
                       height=int(game.h//BLOCK_SIZE),
                       action_dim=3,
                       model_type=model_type,
                       checkpoint=checkpoint)

    # Check if MPS is available and set device accordingly
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Train the agent with Experience Replay Buffer
    batch_size = 10
    num_episodes = 100000
    mean_loss_per_episode = []
    for episode in range(agent.episode_at_last_checkpoint, num_episodes):
        # Store the losses for the current episode
        current_mean_losses = []
        # Start new episode
        game.reset()
        total_reward = 0
        done = False
        # Get current state
        if model_type == 'linear':
            state = game.get_simplified_game_state()
        elif model_type == 'cnn':
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
            if not done and model_type == 'linear':
                next_state = game.get_simplified_game_state()
                next_state = next_state.to(device)
            elif not done and model_type == 'cnn':
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
            minibatch_loss_values = agent.replay(batch_size)
            # Compute mean losses for episode that have losses
            if minibatch_loss_values is not None:
                current_mean_losses = current_mean_losses + [statistics.mean(minibatch_loss_values)]

        if len(current_mean_losses) > 0:
            current_episode_mean_loss = statistics.mean(current_mean_losses)
            mean_loss_per_episode = mean_loss_per_episode + [current_episode_mean_loss]
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Episode loss: {current_episode_mean_loss}")
            # Save model every x episodes
            if episode % 100 == 0:
                save_rl_model(agent.model, agent.optimizer, episode, agent.memory, agent.epsilon)
        else:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# include a main function, to be able to run the script, for notebooks we can comment this out
if __name__ == "__main__":
    playGame(model_type='cnn', checkpoint='SnakeRL_model.pth')
