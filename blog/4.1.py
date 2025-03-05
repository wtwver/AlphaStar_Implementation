import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# Define the Policy class (adjust if your original was different)
class Policy(nn.Module):
    """
    Implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        # Input size changed to 8 for LunarLander's state space
        self.affine1 = nn.Linear(8, 128)

        # Actor's layer: output size changed to 4 for LunarLander's action space
        self.action_head = nn.Linear(128, 4)

        # Critic's layer
        self.value_head = nn.Linear(128, 1)

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        Forward pass of both actor and critic
        """
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_prob, state_values

# Create the environment
env = gym.make("LunarLander-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.4)

# Load the full model
model = torch.load('lunar.pt', weights_only=False, map_location=device)
model = model.to(device)
model.eval()

# Set your save path
save_path = './expert_data/'  # Replace with your path
import os
os.makedirs(save_path, exist_ok=True)

# Lists to store expert data
episode_list, action_list, reward_list, done_list, step_list, state_list, next_state_list = [], [], [], [], [], [], []

for i_episode in range(10000):
    observation = env.reset()[0]  # Take first element of tuple
    observation = torch.FloatTensor(observation).to(device)
    total_step, reward_sum = 0, 0
    
    while True:
        total_step += 1

        observation = observation.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_probs, _ = model(observation)  # Unpack both outputs from the tuple
        action = torch.argmax(action_probs).item()

        observation_1, reward, done, truncated, info = env.step(action)
        observation_1 = torch.FloatTensor(observation_1).to(device)
        
        reward_sum += reward

        # Store data
        episode_list.append(i_episode)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done or truncated)
        step_list.append(total_step)
        state_list.append(observation.cpu().numpy())
        next_state_list.append(observation_1.cpu().numpy())

        observation = observation_1
        
        if done or truncated:
            print("Total reward: {:.2f}, Total step: {}".format(reward_sum, total_step))
            
            print("Saving data")
            save_data = {
                'episode': i_episode,
                'step': step_list,
                'state': state_list,
                'next_state': next_state_list,
                'action': action_list,
                'reward': reward_list,
                'done': done_list
            }

            save_file = '/data_' + str(i_episode)
            path_npy = save_path + save_file + '.npy'
            np.save(path_npy, save_data, allow_pickle=True)

            # Reset lists
            episode_list, action_list, reward_list, done_list, step_list, state_list, next_state_list = [], [], [], [], [], [], []
            break

env.close()