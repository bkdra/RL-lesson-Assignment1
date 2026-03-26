import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters (you should experiment with these options!)
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10  # Update target network every N episodes

# Create environment
env = gym.make('LunarLander-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# TODO: Implement the classes of your agent described in Part B

# Training loop
num_episodes = 1000
rewards_history = []
epsilon = EPSILON_START

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # TODO: select an action
        pass
    
    # TODO: Decay epsilon
    # TODO: Track and log statistics
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

# Testing
# TODO: Test your trained agent

env.close()