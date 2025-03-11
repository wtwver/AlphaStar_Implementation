import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import random
import numpy as np
from typing import Tuple  # Import Tuple for type hints

class Policy(nn.Module):
    """
    Implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        # Input size: 8 for LunarLander's state space
        self.affine1 = nn.Linear(8, 128)

        # Actor's layer: 4 for LunarLander's action space
        self.action_head = nn.Linear(128, 4)

        # Critic's layer
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        Forward pass of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # Actor: raw logits for actions (no softmax here)
        action_logits = self.action_head(x)

        # Critic: evaluates the state
        state_values = self.value_head(x)

        return action_logits, state_values

model = Policy()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
max_episodes = 5000

def compute_supervised_loss(action_logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Computes the supervised loss."""
    # CrossEntropyLoss expects logits, not probabilities

    action_logits = action_logits.squeeze(1)  # Shape changes from [258, 1, 4] to [258, 4]
    action_loss = cce_loss(action_logits, actions)
    return action_loss

def run_supervised_episode(model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a single episode to collect training data."""
    action_logits_list = []
    actions_list = []

    file_list = [d for d in os.listdir('./expert_data') if d.endswith('.npy')]
    file = './expert_data/' + random.choice(file_list)

    data = np.load(file, allow_pickle=True)
    data = np.reshape(data, 1)
    data = data[0]

    state_list = data['state']
    action_list = data['action']
    reward_list = data['reward']

    reward_sum = 0
    model.eval()
    with torch.no_grad():
        for t in range(0, len(state_list) - 1):
            state = torch.from_numpy(state_list[t]).float()
            state = state.unsqueeze(0)

            action_logits_t, value = model(state)

            # Append raw logits, not probabilities
            action_logits_list.append(action_logits_t[0])

            # Ensure action is a long tensor
            actions_list.append(torch.tensor(action_list[t], dtype=torch.long))

            reward_sum += reward_list[t]

    action_logits = torch.stack(action_logits_list)  # Shape: [sequence_length, 4]
    actions = torch.stack(actions_list)              # Shape: [sequence_length]

    return action_logits, actions

def train_supervised_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """Runs a model training step."""
    model.train()
    optimizer.zero_grad()

    # Run the model for one episode to collect training data
    action_logits, actions = run_supervised_episode(model)
    action_logits = action_logits.squeeze(1)  # Shape changes from [258, 1, 4] to [258, 4]

    cce_loss = torch.nn.CrossEntropyLoss()
    loss = cce_loss(action_logits, actions)

    # Add regularization loss if desired
    regularization_loss = torch.tensor(0.0, device=action_logits.device)
    for param in model.parameters():
        if param.requires_grad:
            regularization_loss += torch.norm(param)  # L2 regularization

    total_loss = loss + regularization_loss
    total_loss.backward()
    optimizer.step()

    return total_loss

# Training loop
with tqdm.trange(max_episodes) as t:
    for i in t:
        loss = train_supervised_step(model, optimizer)
        if i % 100 == 0:
            torch.save(model.state_dict(), "LunarLander_SL_Model.pth")
            print("loss: ", loss.item())