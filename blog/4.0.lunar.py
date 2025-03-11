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

# Parser setup
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# Initialize LunarLander environment
env = gym.make('LunarLander-v3')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

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

        # Actor: chooses action by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # Critic: evaluates the state
        state_values = self.value_head(x)

        return action_prob, state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

def main():
    running_reward = 10
    # Define solved threshold for LunarLander since env.spec.reward_threshold is not set
    solved_threshold = 200

    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # Use solved_threshold instead of env.spec.reward_threshold
        if running_reward > solved_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            torch.save(model, 'lunar.pt')
            break

if __name__ == '__main__':
    main()