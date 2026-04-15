import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import random
from utils import plot_training_curves, record_episodes

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 20  # Update target network every N episodes
CHECKPOINT_FREQ = 60  # Save checkpoint every N episodes
RENDER_MODE = 'human'
SOLVED_THRESHOLD = 200.0  # Environment considered solved if mean reward >= this
DEPTH = 2  # Number of hidden layers in the DQN (2, 3, or 4)
HIDDEN_DIM = 128  # Number of units in hidden layers

MODEL_PATH_RECORD = "models/lunar_lander_dqn_depth2.pth"
OUTPUT_DIR_RECORD = "gifs_B/depth2"

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK CLASS
# ══════════════════════════════════════════════════════════════════════════════
class DQN(nn.Module):
    """Deep Q-Network for LunarLander."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, depth: int = 3):
        """
        Initialize the DQN.
        
        Parameters
        ----------
        state_dim : int
            Dimension of the state space (8 for LunarLander).
        action_dim : int
            Dimension of the action space (4 for LunarLander).
        hidden_dim : int
            Size of hidden layers.
        depth : int
            Number of hidden layers.
        """
        super(DQN, self).__init__()

        if depth == 2:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        elif depth == 3:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        elif depth == 4:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            raise ValueError("Unsupported depth. Choose from {2, 3, 4}.")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: return Q-values for actions."""
        return self.net(state)


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER CLASS
# ══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        
        Returns
        -------
        tuple
            Batches of (states, actions, rewards, next_states, dones).
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.from_numpy(np.array(states)).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# DQN AGENT CLASS
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    """Deep Q-Learning agent for LunarLander."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 5e-4, hidden_dim: int = 128, depth: int = 3):
        """
        Initialize the DQN agent.
        
        Parameters
        ----------
        state_dim : int
            Dimension of the state space.
        action_dim : int
            Dimension of the action space.
        learning_rate : float
            Learning rate for the optimizer.
        hidden_dim : int
            Size of hidden layers.
        depth : int
            Number of hidden layers.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-networks
        self.q_network = DQN(state_dim, action_dim, hidden_dim, depth).to(device)
        self.target_network = DQN(state_dim, action_dim, hidden_dim, depth).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state.
        epsilon : float
            Exploration rate.
        
        Returns
        -------
        int
            Selected action.
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def update(self, batch_size: int) -> float:
        """
        Perform a single training step on a batch of transitions.
        
        Parameters
        ----------
        batch_size : int
            Size of the batch to sample.
        
        Returns
        -------
        float
            Loss value for this update.
        """
        if len(self.replay_buffer) < batch_size:
            return float('nan')
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Compute current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)
        
        # Compute Huber loss
        loss = nn.HuberLoss()(q_values, target_q_values)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self) -> None:
        """Update target network weights from Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str) -> None:
        """Save the Q-network to a file."""
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load the Q-network from a file."""
        self.q_network.load_state_dict(torch.load(path, map_location=device))
        self.target_network.load_state_dict(self.q_network.state_dict())


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_dqn(num_episodes: int = 1000, render: bool = False, hidden_dim: int = 128, depth: int = 3, checkpoint_folder: str = "checkpoints"):
    """
    Train the DQN agent on LunarLander.
    
    Parameters
    ----------
    num_episodes : int
        Number of episodes to train.
    render : bool
        Whether to render the environment.
    
    Returns
    -------
    tuple
        (rewards_history, losses_history, epsilon_history, avg_q_history, solved_episode, agent)
    """
    # Create environment
    render_mode = RENDER_MODE if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize agent
    agent = DQNAgent(state_dim, action_dim, learning_rate=LEARNING_RATE, hidden_dim=hidden_dim, depth=depth)
    
    # Training metrics
    rewards_history = []
    losses_history = []
    epsilon_history = []
    avg_q_history = []
    epsilon = EPSILON_START
    running_reward = deque(maxlen=100)
    solved_episode = None
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        num_steps = 0
        episode_q_sum = 0.0
        num_q_steps = 0
        done = False
        
        while not done:
            # Track average max-Q estimate for visited states in this episode.
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                state_q_values = agent.q_network(state_tensor)
                episode_q_sum += state_q_values.max(dim=1)[0].item()
                num_q_steps += 1

            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train on a batch
            loss = agent.update(BATCH_SIZE)
            if not np.isnan(loss):
                episode_loss += loss
                num_steps += 1
            
            episode_reward += reward
            state = next_state
        
        # Update target network
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Track metrics
        rewards_history.append(episode_reward)
        running_reward.append(episode_reward)
        epsilon_history.append(epsilon)
        if num_q_steps > 0:
            avg_q_history.append(episode_q_sum / num_q_steps)
        else:
            avg_q_history.append(float('nan'))
        if num_steps > 0:
            losses_history.append(episode_loss / num_steps)
        else:
            losses_history.append(float('nan'))

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Save periodic checkpoint
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            os.makedirs("checkpoints/"+checkpoint_folder, exist_ok=True)
            checkpoint_path = f"checkpoints/{checkpoint_folder}/lunar_lander_dqn_ep{episode + 1}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Logging
        if (episode + 1) % 50 == 0:
            mean_reward = np.mean(running_reward)
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Mean(100): {mean_reward:7.2f} | "
                  f"Epsilon: {epsilon:.3f}")
        
        # Check if solved
        if solved_episode is None and len(running_reward) == 100:
            mean_reward = np.mean(running_reward)
            if mean_reward >= SOLVED_THRESHOLD:
                solved_episode = episode + 1
                print(f"\n✓ Environment SOLVED at episode {solved_episode}!")
                print(f"  Mean reward (100 episodes): {mean_reward:.2f}\n")
    
    env.close()
    
    return rewards_history, losses_history, epsilon_history, avg_q_history, solved_episode, agent


# ══════════════════════════════════════════════════════════════════════════════
# TESTING/EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def test_agent(model_path: str = "lunar_lander_dqn.pth", num_episodes: int = 10, render: bool = True, hidden_dim: int = 128):
    """
    Load a trained agent and evaluate it on the environment.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    num_episodes : int
        Number of episodes to evaluate.
    render : bool
        Whether to render the environment.
    
    Returns
    -------
    list
        Rewards for each episode.
    """
    # Create environment
    render_mode = RENDER_MODE if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_dim, action_dim, hidden_dim=hidden_dim)
    
    # Load trained weights
    agent.load(model_path)
    print(f"Loaded model from '{model_path}'")
    
    # Set network to evaluation mode
    agent.q_network.eval()
    agent.target_network.eval()
    
    # Evaluate
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Use greedy policy (epsilon=0 for pure exploitation)
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d} | Reward: {episode_reward:7.2f}")
    
    env.close()
    
    # Print statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n{'='*50}")
    print(f"Evaluation Statistics ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:  {min(episode_rewards):.2f}")
    print(f"Max Reward:  {max(episode_rewards):.2f}")
    print(f"{'='*50}\n")
    
    return episode_rewards

def record_test_agent(model_path: str = "lunar_lander_dqn.pth", output_dir: str = "gifs_B", num_episodes: int = 10, render: bool = True, hidden_dim: int = 128, depth: int = 3):
    """
    Load a trained agent and evaluate it on the environment.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    num_episodes : int
        Number of episodes to evaluate.
    render : bool
        Whether to render the environment.
    hidden_dim : int
        Number of units in hidden layers.
    depth : int
        Number of hidden layers.

    Returns
    -------
    list
        Rewards for each episode.
    """
    # Create environment
    render_mode = RENDER_MODE if render else None
    env = gym.make('LunarLander-v3', render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_dim, action_dim, hidden_dim=hidden_dim, depth=depth)
    
    # Load trained weights
    agent.load(model_path)
    print(f"Loaded model from '{model_path}'")
    
    # Set network to evaluation mode
    agent.q_network.eval()
    agent.target_network.eval()

    def policy(state):
        """Policy function for recording episodes."""
        action = agent.select_action(state, epsilon=0.0)  # Greedy action selection
        return action
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Evaluate
    record_episodes(num_episodes, output_dir, policy)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    
    # Check command-line argument
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Load and test the trained model
        print("Loading and testing trained agent...\n")
        test_agent(model_path="lunar_lander_dqn.pth", num_episodes=10, render=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "record_gifs":
        # Record episodes using a random policy
        print("Recording episodes with a random policy...\n")
        record_test_agent(model_path=MODEL_PATH_RECORD, output_dir=OUTPUT_DIR_RECORD, num_episodes=5, render=True, hidden_dim=HIDDEN_DIM, depth=DEPTH)
    else:
        # Train the agent
        print("Starting DQN training on LunarLander-v3\n")
        rewards, losses, epsilons, avg_q_values, solved_at, agent = train_dqn(num_episodes=600, render=False, hidden_dim=HIDDEN_DIM, checkpoint_folder=f"depth{DEPTH}", depth = DEPTH)
        
        # Save the trained model
        agent.save(f"lunar_lander_dqn_depth{DEPTH}.pth")
        print(f"\nModel saved to 'lunar_lander_dqn_depth{DEPTH}.pth'")

        metrics = {
            'episode_rewards': rewards,
            'avg_losses': losses,
            'epsilons': epsilons,
            'mean_q_values': avg_q_values,
            'solved_at': solved_at,
        }
        plot_training_curves(metrics, out_dir="outputs/part_b_c_2", fileName_suffix=f"_depth{DEPTH}")