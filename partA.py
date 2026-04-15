import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from utils import print_stats, make_env_with_video, plot_baseline

# Hyperparameters (you should experiment with these options!)
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes
RENDER_MODE = 'human'

# Create environment
# env = gym.make('LunarLander-v3', render_mode=RENDER_MODE)
env = make_env_with_video()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# TODO: Implement the classes of your agent described in Part B

# Training loop
num_episodes = 1000
rewards_history = []
episode_lengths = []
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_length += 1
    
    # TODO: Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_history.append(episode_reward)
    episode_lengths.append(episode_length)
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

stats = {
    'mean_reward': float(np.mean(rewards_history)),
    'std_reward': float(np.std(rewards_history)),
    'min_reward': float(np.min(rewards_history)),
    'max_reward': float(np.max(rewards_history)),
    'mean_length': float(np.mean(episode_lengths)),
    'success_rate': float(np.mean(np.array(rewards_history) >= 200.0)),
}

plot_stats = {
    'episode_rewards': rewards_history,
    'episode_lengths': episode_lengths,
    'mean_reward': float(np.mean(rewards_history)),
    'std_reward': float(np.std(rewards_history)),
    'min_reward': float(np.min(rewards_history)),
    'max_reward': float(np.max(rewards_history)),
    'mean_length': float(np.mean(episode_lengths)),
    'success_rate': float(np.mean(np.array(rewards_history) >= 200.0)),
}


print_stats(stats)
plot_baseline(plot_stats)

# Testing
# TODO: Test your trained agent

env.close()