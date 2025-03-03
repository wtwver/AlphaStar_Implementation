import os
import argparse
import random
import numpy as np
from collections import defaultdict
import gzip
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# pysc2-related imports (if still needed for actions / features)
from pysc2.lib import actions, features
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES as ACTION_TYPES

import utils  # assume your preprocessing functions are available

import network  # assume network.make_model returns a torch.nn.Module

parser = argparse.ArgumentParser(description='AlphaStar SL in PyTorch')
parser.add_argument('--environment', type=str, default='MoveToBeacon', help='name of SC2 environment')
parser.add_argument('--workspace_path', type=str, required=True, help='root directory for checkpoint storage')
parser.add_argument('--visualize', type=bool, default=False, help='render with pygame')
parser.add_argument('--model_name', type=str, default='fullyconv', help='model name')
parser.add_argument('--training', type=bool, default=False, help='training model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')
parser.add_argument('--seed', type=int, default=123, help='seed number')
parser.add_argument('--training_episode', type=int, default=5000, help='training number')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--save', type=bool, default=False, help='save trained model')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--player_1', type=str, default='terran', help='race of player 1')
parser.add_argument('--player_2', type=str, default='terran', help='race of player 2')
parser.add_argument('--screen_size', type=int, default=32, help='screen resolution')
parser.add_argument('--minimap_size', type=int, default=32, help='minimap resolution')
parser.add_argument('--replay_dir', type=str, default="replay", help='replay save path')
parser.add_argument('--replay_hkl_file_path', type=str, default="replay", help='path of replay file for SL')
parser.add_argument('--save_replay_episodes', type=int, default=10, help='extra info (if needed)')
parser.add_argument('--tensorboard_path', type=str, default="tensorboard", help='Folder for saving Tensorboard log file')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Device configuration
device = torch.device("mps" if args.gpu_use and torch.backends.mps.is_available() else "cuda" if args.gpu_use and torch.cuda.is_available() else "cpu")
print(device)
if args.gpu_use and torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Some pysc2 constants and helper functions
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_NUM_FUNCTIONS = len(actions.FUNCTIONS)

# Determine spatial actions
is_spatial_action = {}
for name, arg_type in actions.TYPES._asdict().items():
    # HACK: we should infer the point type automatically
    is_spatial_action[arg_type] = name in ['minimap', 'screen', 'screen2']

def check_nonzero(mask):
    # mask is a 2D numpy array of booleans
    indices = np.argwhere(mask)
    for idx in indices:
        x, y = idx[1], idx[0]
        # do something if needed
        pass

def take_vector_elements(vectors, indices):
    # vectors: Tensor of shape [B, ...]
    # indices: Tensor of shape [B]
    # Return one element per batch by gathering along last dimension
    batch_indices = torch.arange(vectors.size(0), device=vectors.device)
    return vectors[batch_indices, indices]

def actions_to_pysc2(fn_id, arg_ids, size):
    height, width = size
    actions_list = []
    a_0 = int(fn_id)
    arg_list = []
    for arg_type in FUNCTIONS._func_list[a_0].args:
        a_id = int(arg_ids[arg_type])
        if is_spatial_action[arg_type]:
            arg = [a_id % width, a_id // height]
        else:
            arg = [a_id]
        arg_list.append(arg)
    action = FunctionCall(a_0, arg_list)
    actions_list.append(action)
    return actions_list

def mask_unused_argument_samples(fn_id, arg_ids):
    args_out = dict()
    for arg_type in actions.TYPES:
        args_out[arg_type] = arg_ids[arg_type][0]
    a_0 = fn_id[0]
    unused_types = set(ACTION_TYPES) - set(FUNCTIONS._func_list[int(a_0)].args)
    for arg_type in unused_types:
        args_out[arg_type] = -1
    return fn_id, args_out

def mask_unavailable_actions(available_actions, fn_logits):
    # available_actions: Tensor [B, num_fn] assumed to be 0/1
    available_actions = available_actions.float()
    # Mask logits: set logits to a very large negative value where action is unavailable.
    mask = available_actions > 0
    masked_logits = fn_logits.clone()
    masked_logits[~mask] = -1e9
    # Optionally normalize (softmax later in loss if you want probability interpretation)
    return masked_logits

def sample_actions(available_actions, fn_logits, arg_logits_dict):
    # sample a function id and one argument per arg type
    masked_fn_logits = mask_unavailable_actions(available_actions, fn_logits)
    fn_dist = torch.distributions.Categorical(logits=masked_fn_logits)
    fn_sample = fn_dist.sample()

    arg_samples = {}
    for arg_type, logits in arg_logits_dict.items():
        arg_dist = torch.distributions.Categorical(logits=logits)
        arg_samples[arg_type] = arg_dist.sample()
    return fn_sample, arg_samples

def compute_policy_entropy(available_actions, fn_logits, arg_logits_dict, fn_ids, arg_ids):
    # Compute the entropy of the policy (for logging, for example)
    # Here we apply the same masking as in sample_actions then compute entropy.
    masked_fn_logits = mask_unavailable_actions(available_actions, fn_logits)
    fn_prob = torch.softmax(masked_fn_logits, dim=1)
    entropy = -torch.sum(fn_prob * torch.log(fn_prob + 1e-10), dim=1).mean()

    # Add entropy from arguments (each arg type separately)
    for index, arg_type in enumerate(actions.TYPES):
        logits = arg_logits_dict[arg_type]
        arg_prob = torch.softmax(logits, dim=1)
        batch_mask = (arg_ids[:, index] != -1).float().unsqueeze(1)
        arg_entropy = -torch.sum(arg_prob * torch.log(arg_prob + 1e-10), dim=1)
        # Average only over valid samples
        if batch_mask.sum() > 0:
            entropy += (arg_entropy * batch_mask.squeeze(1)).sum() / batch_mask.sum()
    return entropy

# -------------------------------
# Create a PyTorch Dataset to yield trajectories.
# -------------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, num_trajectories=1000):
        super().__init__()
        self.num_trajectories = num_trajectories
        # The replay file pattern is assumed to be: replay_hkl_file_path + '*.hkl'
        self.replay_pattern = args.replay_hkl_file_path + '*.hkl'
    
    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        replay_files = [ f for f in os.listdir(args.replay_hkl_file_path) if f.endswith('.pkl') ]
        replay_file = random.choice(replay_files)
        
        try:
            with gzip.open(args.replay_hkl_file_path + replay_file) as f:
                replay = pickle.load(f)
        except Exception as e:
            print(e)
            # If failed, pick another sample recursively (or return zeros)
            return self.__getitem__(idx)

        # Prepare lists for each feature in the trajectory.
        # (Here we follow the structure of the TF generator.)
        feature_screen_list = []
        feature_minimap_list = []
        player_list = []
        feature_units_list = []
        available_actions_list = []
        fn_id_list = []
        args_ids_list = []
        game_loop_list = []
        last_action_type_list = []
        build_queue_list = []
        single_select_list = []
        multi_select_list = []
        score_cumulative_list = []
        last_action_type = [0]

        replay_file_length = len(replay['home_game_loop'])
        # We go through one replay (note: you might wish to sample a fixed-length segment)
        for sample_idx in range(1, replay_file_length):
            # Preprocess different features (the utils functions are assumed to be available)
            fs = utils.preprocess_screen(torch.tensor(replay['home_feature_screen'][sample_idx-1]))
            fs = np.transpose(fs, (1, 2, 0))
            feature_screen_list.append(fs)

            fm = utils.preprocess_minimap(torch.tensor(replay['home_feature_minimap'][sample_idx-1]))
            fm = np.transpose(fm, (1, 2, 0))
            feature_minimap_list.append(fm)

            pl = utils.preprocess_player(replay['home_player'][sample_idx-1])
            player_list.append(pl)

            fu = utils.preprocess_feature_units(replay['home_feature_units'][sample_idx-1], args.screen_size)
            feature_units_list.append(fu)

            ga = replay['home_game_loop'][sample_idx-1]
            game_loop_list.append(ga)

            aa = utils.preprocess_available_actions(replay['home_available_actions'][sample_idx-1])
            available_actions_list.append(aa)

            bq = utils.preprocess_build_queue(replay['home_build_queue'][sample_idx-1])
            build_queue_list.append(bq)

            ss = utils.preprocess_single_select(replay['home_single_select'][sample_idx-1])
            single_select_list.append(ss)

            ms = utils.preprocess_multi_select(replay['home_multi_select'][sample_idx-1])
            multi_select_list.append(ms)

            sc = utils.preprocess_score_cumulative(replay['home_score_cumulative'][sample_idx-1])
            score_cumulative_list.append(sc)

            # Get a random action from this timestep’s actions
            actions_candidates = replay['home_action'][sample_idx]
            action_chosen = random.choice(actions_candidates)
            fn_id = int(action_chosen[0])
            fn_id_list.append(fn_id)
            
            # Build argument ids list for this time step.
            arg_ids = []
            args_temp = {arg_type: -1 for arg_type in actions.TYPES}
            arg_index = 0
            for arg_type in FUNCTIONS._func_list[fn_id].args:
                args_temp[arg_type] = action_chosen[1][arg_index]
                arg_index += 1
            for arg_type in actions.TYPES:
                aid = args_temp[arg_type]
                if isinstance(aid, list):
                    if len(aid) == 2:
                        aid = aid[0] + aid[1] * args.screen_size
                    else:
                        aid = int(aid[0])
                arg_ids.append(aid)
            args_ids_list.append(arg_ids)

            # Update last action type for next iteration.
            last_action_type_list.append(last_action_type)
            last_action_type = [fn_id]

            # For simplicity, we break after completing one trajectory
            if sample_idx == replay_file_length - 1:
                break
        
        # Convert lists into numpy arrays. (You might wish to pad/clip if sequences are variable-length.)
        sample = {
            "feature_screen": np.array(feature_screen_list, dtype=np.float32),
            "feature_minimap": np.array(feature_minimap_list, dtype=np.float32),
            "player": np.array(player_list, dtype=np.float32),
            "feature_units": np.array(feature_units_list, dtype=np.float32),
            "available_actions": np.array(available_actions_list, dtype=np.float32),
            "fn_ids": np.array(fn_id_list, dtype=np.int64),  # long tensor for indices
            "args_ids": np.array(args_ids_list, dtype=np.int64),
            "game_loop": np.array(game_loop_list, dtype=np.float32),
            "last_action_type": np.array(last_action_type_list, dtype=np.int64),
            "build_queue": np.array(build_queue_list, dtype=np.float32),
            "single_select": np.array(single_select_list, dtype=np.float32),
            "multi_select": np.array(multi_select_list, dtype=np.float32),
            "score_cumulative": np.array(score_cumulative_list, dtype=np.float32)
        }
        return sample

# Create the dataset and dataloader.
dataset = TrajectoryDataset(num_trajectories=1000)
# Here we set batch_size=1 because each sample is already a full trajectory.
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

model = network.make_model(args.model_name)
model.to(device)

if args.pretrained_model is not None:
    model.load_state_dict(torch.load(os.path.join(args.workspace_path, "Models", args.pretrained_model)))

writer = SummaryWriter(log_dir=os.path.join(args.workspace_path, "tensorboard", "supervised_learning"))
criterion = nn.CrossEntropyLoss()
criterion_args = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
arg_types_list = list(actions.TYPES)

def supervised_replay(batch_sample, memory_state, carry_state):
    """
    Process one trajectory segment.
    All inputs are torch tensors on device.
    Assumes trajectory time length T; the inputs come with shape [T, ...]
    """
    T = batch_sample["feature_screen"].shape[0]
    # We will save predictions at each time step.
    fn_logits_list = []
    arg_logits_dict = {arg: [] for arg in arg_types_list}

    # Process the trajectory time step by time step.
    for t in range(T):
        print(batch_sample["feature_screen"][t].shape) if debug else None

        input_dict = {
            "feature_screen": batch_sample["feature_screen"][t].unsqueeze(0),  
            "feature_minimap": batch_sample["feature_minimap"][t].unsqueeze(0),
            "player": batch_sample["player"][t].unsqueeze(0),
            "feature_units": batch_sample["feature_units"][t].unsqueeze(0),
            "memory_state": memory_state,
            "carry_state": carry_state,
            "game_loop": batch_sample["game_loop"][t].unsqueeze(0),
            "available_actions": batch_sample["available_actions"][t].unsqueeze(0),
            "act_history": batch_sample["last_action_type"][t].unsqueeze(0),
            "build_queue": batch_sample["build_queue"][t].unsqueeze(0),
            "single_select": batch_sample["single_select"][t].unsqueeze(0),
            "multi_select": batch_sample["multi_select"][t].unsqueeze(0),
            "score_cumulative": batch_sample["score_cumulative"][t].unsqueeze(0)
        }
        # Forward pass.
        # It's assumed that the model returns a dict with keys:
        # 'fn_out': [1, num_fn] logits,
        # 'args_out': a dict mapping each arg_type to logits [1, arg_dim],
        # 'final_memory_state', 'final_carry_state'
        output = model(**input_dict)
        for i, o in enumerate(output):
            print("output", i, len(o)) if debug else None

        fn_logits = output[0]  # shape [1, num_fn]
        args_out = output[1]  # dict mapping argument type -> [1, arg_size]
        memory_state = output[3]
        carry_state = output[4]

        fn_logits_list.append(fn_logits.squeeze(0))

        for i, arg in enumerate(arg_types_list):
            print(i, arg) if debug else None
            arg_logits_dict[arg].append(args_out[i].squeeze(0))

    # Stack all predictions: resulting shapes [T, num_fn] and for each arg [T, arg_dim]
    fn_logits_tensor = torch.stack(fn_logits_list, dim=0)
    for arg in arg_types_list:
        arg_logits_dict[arg] = torch.stack(arg_logits_dict[arg], dim=0)

    # Get ground truth tensors.
    # fn_ids: shape [T], already long tensor.
    gt_fn = batch_sample["fn_ids"].to(device)
    # available_actions: shape [T, num_fn]
    available_actions = batch_sample["available_actions"].to(device)
    # For arguments: shape [T, num_args]. We assume the order matches arg_types_list.
    gt_args = batch_sample["args_ids"].to(device)  # shape [T, num_args]

    # Compute function id loss.
    # First, mask unavailable actions.
    masked_fn_logits = mask_unavailable_actions(available_actions, fn_logits_tensor)
    # Flatten the logits and targets along batch and time.
    # New shape: [batch * T, num_fn] and [batch * T] respectively.
    masked_fn_logits_flat = masked_fn_logits.view(-1, masked_fn_logits.size(-1))  # e.g. [8, 573]
    gt_fn_flat = gt_fn.view(-1)  # e.g. [8]
    fn_loss = criterion(masked_fn_logits_flat, gt_fn_flat)
    # print('fn_loss: {:.4f}'.format(fn_loss.item()))
    # Compute argument losses.
    arg_loss = 0
    for idx, arg in enumerate(arg_types_list):
        # gt for current arg is in gt_args[:, :, idx] (shape: [batch, T])
        target = gt_args[:, :, idx].view(-1)
        logits = arg_logits_dict[arg].view(-1, arg_logits_dict[arg].size(-1))
        # print(f'target {target}, logits {logits}')
        if (target != -1).sum().item() > 0:
            tmp = criterion_args(logits, target)
            if tmp is not None and not torch.isnan(tmp):
                arg_loss += tmp

    total_loss = fn_loss + arg_loss
    return total_loss, memory_state, carry_state

def supervised_train(dataloader, training_episodes):
    training_step = 0
    # You can choose a fixed step length for unrolling an episode.
    step_length = 8
    model.train()
    for epoch in tqdm(range(training_episodes), desc='Training epochs'):
        print("Epoch: {}".format(epoch))
        for sample in dataloader:
            # sample is a dict with numpy arrays; convert them to torch tensors.
            batch_sample = {
                key: torch.tensor(value.clone().detach(), device=device)
                for key, value in sample.items()
            }
            # Get the trajectory length, assume it is the first dimension.
            T_total = batch_sample["feature_screen"].shape[1]
            # Initialize latent states (assume dimensions [1, state_dim])
            memory_state = torch.zeros(1, 1024, device=device)
            carry_state = torch.zeros(1, 1024, device=device)
            # Unroll over the trajectory in chunks of step_length.
            for t in range(0, T_total, step_length):
                # Make sure we have a full step_length (skip incomplete segments).
                print('t: {}, step_length: {}, T_total: {}'.format(t, step_length, T_total)) if debug else None
                if t + step_length > T_total:
                    print('t + step_length > T_total') if debug else None
                    break
                # Slice the segment.
                segment = { key: batch_sample[key][:, t:t+step_length] for key in batch_sample }
                optimizer.zero_grad()
                loss, memory_state, carry_state = supervised_replay(segment, memory_state, carry_state)
                loss.backward()
                optimizer.step()

                training_step += 1
                print("training_step: {} loss: {:.4f}".format(training_step, loss.item())) if debug else None
                if training_step % 250 == 0:
                    writer.add_scalar("total_loss", loss.item(), training_step)
                    print("loss: {:.4f}".format(loss.item()))
                if training_step % 5000 == 0:
                    save_path = os.path.join(args.workspace_path, "Models", f"supervised_model_{training_step}")
                    torch.save(model.state_dict(), save_path)
                # (Optional) free GPU memory, if necessary.
                torch.cuda.empty_cache()

def main():
    supervised_train(dataloader, args.training_episode)

if __name__ == "__main__":
    debug = 0
    main()