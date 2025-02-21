from pysc2.lib import actions, features, units
import torch
import math
from collections import namedtuple
import os
import random
import collections
import threading
import time
import timeit
from absl import logging

# Constants from pysc2 features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_NUM_FUNCTIONS = len(actions.FUNCTIONS)

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCREEN_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_VISIBILITY_MAP = features.SCREEN_FEATURES.visibility_map.index

_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_MINIMAP_CAMERA = features.MINIMAP_FEATURES.camera.index
_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index

all_unit_list = [0, 37, 45, 48, 317, 21, 341, 342, 18, 27, 132, 20, 5, 47, 21, 
                 19, 483, 51, 28, 42, 53, 268, 472, 49, 41, 830, 105, 9, 1680, 110]

# Marine = 48
# Zergling = 105
# Baneling = 9
# Roach = 110
# Mineral = 1680
# Beacon = 317

essential_unit_list = [0, 45, 48, 317, 21, 341, 18, 27, 20, 19, 483, 500] # For Simple64
#essential_unit_list = [0, 48, 105, 9]  # For Minigame
# essential_unit_list = [0, 48, 1680]



def preprocess_screen(screen):
    """
    Preprocess the screen tensor using PyTorch.
    
    Args:
        screen (torch.Tensor): Input screen tensor with shape (C, H, W), 
                               where C == len(features.SCREEN_FEATURES).
        
    Returns:
        torch.Tensor: Preprocessed screen tensor.
    """
    layers = []
    assert screen.shape[0] == len(features.SCREEN_FEATURES)
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_UNIT_TYPE:
            scale = len(essential_unit_list)
            layer = torch.zeros((scale, screen.shape[1], screen.shape[2]), 
                                dtype=torch.float32, device=screen.device)
            for j in range(len(all_unit_list)):
                mask = (screen[i] == all_unit_list[j])
                indices = mask.nonzero()
                if len(indices) > 2:
                    if all_unit_list[j] in essential_unit_list:
                        unit_index = essential_unit_list.index(all_unit_list[j])
                        layer[unit_index, indices[0], indices[1]] = 1.0
                    else:
                        layer[-1, indices[0], indices[1]] = 1.0
            layers.append(layer)
        elif i == _SCREEN_SELECTED:
            scale_val = features.SCREEN_FEATURES[i].scale
            layer = torch.zeros((scale_val, screen.shape[1], screen.shape[2]), 
                                dtype=torch.float32, device=screen.device)
            for j in range(scale_val):
                mask = (screen[i] == j)
                indices = mask.nonzero()
                if len(indices) >= 2 :
                    layer[j, indices[0], indices[1]] = 1.0
            layers.append(layer)
        elif i == _SCREEN_UNIT_HIT_POINTS:
            scale_tensor = torch.tensor(features.SCREEN_FEATURES[i].scale, 
                                        dtype=torch.float32, device=screen.device)
            layer = torch.log(screen[i:i+1] + 1) / torch.log(scale_tensor)
            layers.append(layer)
        else:
            # For other features you might want to change the preprocessing.
            layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    return torch.cat(layers, dim=0)


def preprocess_minimap(minimap):
    """
    Preprocess the minimap tensor using PyTorch.
    
    Args:
        minimap (torch.Tensor): Input minimap tensor with shape (C, H, W),
                                where C == len(features.MINIMAP_FEATURES).
        
    Returns:
        torch.Tensor: Preprocessed minimap tensor.
    """
    layers = []
    assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == features.FeatureType.SCALAR:
            layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
        elif i == _MINIMAP_CAMERA or i == _MINIMAP_PLAYER_RELATIVE:
            scale_val = features.MINIMAP_FEATURES[i].scale
            layer = torch.zeros((scale_val, minimap.shape[1], minimap.shape[2]), 
                                dtype=torch.float32, device=minimap.device)
            for j in range(scale_val):
                mask = (minimap[i] == j)
                indices = mask.nonzero()
                if len(indices) > 2:
                    layer[j, indices[0], indices[1]] = 1.0
            layers.append(layer)
        else:
            layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    return torch.cat(layers, dim=0)


FlatFeature = namedtuple('FlatFeatures', ['index', 'type', 'scale', 'name'])
FLAT_FEATURES = [
    FlatFeature(0,  features.FeatureType.SCALAR, 1, 'player_id'),
    FlatFeature(1,  features.FeatureType.SCALAR, 10000, 'minerals'),
    FlatFeature(2,  features.FeatureType.SCALAR, 10000, 'vespene'),
    FlatFeature(3,  features.FeatureType.SCALAR, 200, 'food_used'),
    FlatFeature(4,  features.FeatureType.SCALAR, 200, 'food_cap'),
    FlatFeature(5,  features.FeatureType.SCALAR, 200, 'food_army'),
    FlatFeature(6,  features.FeatureType.SCALAR, 200, 'food_workers'),
    FlatFeature(7,  features.FeatureType.SCALAR, 200, 'idle_worker_count'),
    FlatFeature(8,  features.FeatureType.SCALAR, 200, 'army_count'),
    FlatFeature(9,  features.FeatureType.SCALAR, 200, 'warp_gate_count'),
    FlatFeature(10, features.FeatureType.SCALAR, 200, 'larva_count'),
]


def preprocess_player(player):
    """
    Preprocess player flat features using PyTorch.
    
    Args:
        player (list or torch.Tensor): Player features.
    
    Returns:
        torch.Tensor: Preprocessed player features.
    """
    layers = []
    for s in FLAT_FEATURES:
        if s.index in [1, 2]:
            scale_tensor = torch.tensor(s.scale, dtype=torch.float32)
            value = (torch.tensor(player[s.index], dtype=torch.float32)
                     if not torch.is_tensor(player[s.index])
                     else player[s.index])
            out = torch.log(value + 1) / torch.log(scale_tensor)
            layers.append(out)
        else:
            value = (torch.tensor(player[s.index], dtype=torch.float32)
                     if not torch.is_tensor(player[s.index])
                     else player[s.index])
            layers.append(value / s.scale)
    return torch.stack(layers)


def preprocess_available_actions(available_action):
    """
    Preprocess available actions into a one-hot encoded vector.
    
    Args:
        available_action (int or list): Available action index(es).
    
    Returns:
        torch.Tensor: One-hot encoded available actions vector.
    """
    available_actions = torch.zeros(_NUM_FUNCTIONS, dtype=torch.float64)
    available_actions[available_action] = 1.0
    return available_actions


def preprocess_feature_units(feature_units, feature_screen_size):
    """
    Preprocess feature units into a fixed-size tensor of shape (50, 8).
    
    Args:
        feature_units (list): List of feature unit objects.
        feature_screen_size (int): Screen size (unused in this function).
    
    Returns:
        torch.Tensor: Tensor of preprocessed feature units.
    """
    feature_units_list = []
    feature_units_length = len(feature_units)
    for i, feature_unit in enumerate(feature_units):
        unit_features = []
        unit_features.append(feature_unit.unit_type / 2000.0)
        unit_features.append(feature_unit.alliance / 4.0)
        unit_features.append(feature_unit.health / 10000.0)
        unit_features.append(feature_unit.shield / 10000.0)
        unit_features.append(feature_unit.x / 100.0)
        unit_features.append(feature_unit.y / 100.0)
        unit_features.append(float(feature_unit.is_selected))
        unit_features.append(feature_unit.build_progress / 500.0)
        feature_units_list.append(unit_features)
        if i >= 49:
            break
    for i in range(feature_units_length, 50):
        feature_units_list.append([0.0] * 8)
    entity_array = torch.tensor(feature_units_list, dtype=torch.float32)
    return entity_array


SingleSelectFeature = namedtuple('SingleSelectFeature', ['index', 'type', 'scale', 'name'])
SINGLE_SELECT_FEATURES = [
    SingleSelectFeature(0,  features.FeatureType.SCALAR, len(essential_unit_list), 'unit_type'),
    SingleSelectFeature(1,  features.FeatureType.SCALAR, 4, 'player_relative'),
    SingleSelectFeature(2,  features.FeatureType.SCALAR, 2000, 'health'),
]


def preprocess_single_select(single_select):
    """
    Preprocess single select feature.
    
    Args:
        single_select (list): List of single select feature units.
    
    Returns:
        torch.Tensor: Preprocessed single select feature as a tensor.
    """
    if len(single_select) != 0:
        single_select_elem = single_select[0]
        layers = []
        for s in SINGLE_SELECT_FEATURES:
            if s.index == 2:
                scale_tensor = torch.tensor(s.scale, dtype=torch.float32)
                value = (torch.tensor(single_select_elem[s.index], dtype=torch.float32)
                         if not torch.is_tensor(single_select_elem[s.index])
                         else single_select_elem[s.index])
                out = torch.log(value + 1) / torch.log(scale_tensor)
                layers.append(out)
            elif s.index == 0:
                out = essential_unit_list.index(single_select_elem[s.index]) / s.scale
                layers.append(out)
            else:
                out = single_select_elem[s.index] / s.scale
                layers.append(out)
        return torch.tensor(layers, dtype=torch.float32)
    else:
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


ScoreCumulativeFeature = namedtuple('ScoreCumulativeFeature', ['index', 'type', 'scale', 'name'])
SCORE_CUMULATIVE_FEATURES = [
    ScoreCumulativeFeature(0,  features.FeatureType.SCALAR, 25000, ' score '),
    ScoreCumulativeFeature(1,  features.FeatureType.SCALAR, 5000, 'idle_production_time'),
    ScoreCumulativeFeature(2,  features.FeatureType.SCALAR, 10000, 'idle_worker_time'),
    ScoreCumulativeFeature(3,  features.FeatureType.SCALAR, 10000, 'total_value_units'),
    ScoreCumulativeFeature(4,  features.FeatureType.SCALAR, 10000, 'total_value_structures'),
    ScoreCumulativeFeature(5,  features.FeatureType.SCALAR, 10000, 'killed_value_units'),
    ScoreCumulativeFeature(6,  features.FeatureType.SCALAR, 10000, 'killed_value_structures'),
    ScoreCumulativeFeature(7,  features.FeatureType.SCALAR, 10000, 'collected_minerals'),
    ScoreCumulativeFeature(8,  features.FeatureType.SCALAR, 10000, 'collected_vespene'),
    ScoreCumulativeFeature(9,  features.FeatureType.SCALAR, 2000, 'collection_rate_minerals'),
    ScoreCumulativeFeature(10, features.FeatureType.SCALAR, 2000, 'collection_rate_vespene'),
    ScoreCumulativeFeature(11, features.FeatureType.SCALAR, 10000, 'spent_minerals'),
    ScoreCumulativeFeature(12, features.FeatureType.SCALAR, 10000, 'spent_vespene'),
]


def preprocess_score_cumulative(score_cumulative):
    """
    Preprocess score cumulative features.
    
    Args:
        score_cumulative (list or tensor): Score cumulative features.
    
    Returns:
        torch.Tensor: Preprocessed score cumulative features.
    """
    layers = []
    for s in SCORE_CUMULATIVE_FEATURES:
        if s.index in [9, 10]:
            out = score_cumulative[s.index] / s.scale
            layers.append(out)
        else:
            # The original code computes a logarithm but then overwrites it
            out = score_cumulative[s.index] / s.scale
            layers.append(out)
    return torch.tensor(layers, dtype=torch.float32)


def preprocess_build_queue(build_queue):
    """
    Preprocess build queue into a fixed-size tensor.
    
    Args:
        build_queue (list): List representing the build queue.
    
    Returns:
        torch.Tensor: Tensor of shape (5,) with preprocessed build queue.
    """
    build_queue_length = len(build_queue)
    if build_queue_length > 5:
        build_queue_length = 5
    layers = [0.0] * 5
    for i in range(build_queue_length):
        layers[i] = essential_unit_list.index(build_queue[i][0]) / float(len(essential_unit_list))
    return torch.tensor(layers, dtype=torch.float32)


def preprocess_multi_select(multi_select):
    """
    Preprocess multi select feature into a fixed-size tensor.
    
    Args:
        multi_select (list): List of multi select feature units.
    
    Returns:
        torch.Tensor: Tensor of preprocessed multi select features (shape (10,)).
    """
    multi_select_length = len(multi_select)
    if multi_select_length > 10:
        multi_select_length = 10
    layers = [0.0] * 10
    for i in range(multi_select_length):
        layers[i] = essential_unit_list.index(multi_select[i][0]) / float(len(essential_unit_list))
    return torch.tensor(layers, dtype=torch.float32)


def positional_encoding(max_position, embedding_size, add_batch_dim=False):
    """
    Compute positional encodings using PyTorch.
    
    Args:
        max_position (int): Maximum sequence length.
        embedding_size (int): Dimension of embeddings.
        add_batch_dim (bool): Whether to add a batch dimension.
    
    Returns:
        torch.Tensor: Positional encoding tensor.
    """
    positions = torch.arange(max_position, dtype=torch.float32)
    div_term = (2 * (torch.arange(embedding_size, dtype=torch.float32) // 2)) / float(embedding_size)
    angle_rates = 1 / torch.pow(torch.tensor(10000.0), div_term)
    angle_rads = positions.unsqueeze(1) * angle_rates.unsqueeze(0)
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    if add_batch_dim:
        angle_rads = angle_rads.unsqueeze(0)
    return angle_rads