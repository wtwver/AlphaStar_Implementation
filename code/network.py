import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib import actions
# _NUM_FUNCTIONS is defined based on the available actions.
_NUM_FUNCTIONS = len(actions.FUNCTIONS)
debug = 0

class SpatialEncoder(nn.Module):
    def __init__(self, height, width, channel):
        super(SpatialEncoder, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        
        self.network = nn.Sequential(
            nn.LazyConv2d(out_channels=channel, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
            'channel': self.channel
        }

    def forward(self, spatial_feature):
        spatial_feature_encoded = self.network(spatial_feature)
        return spatial_feature_encoded

class FullyConv(nn.Module):
    def __init__(self, screen_size, minimap_size):
        """
        Fully convolutional network version.

        Expected inputs (in PyTorch "channels-first" format):
          - feature_screen: (batch, 24, screen_size, screen_size)
          - feature_minimap: (batch, 7, minimap_size, minimap_size)
          - player: (batch, 11)
          - feature_units: (batch, 50, 8)
          - game_loop: (batch, 1)
          - available_actions: (batch, 573)
          - build_queue: (batch, 5)
          - single_select: (batch, 3)
          - multi_select: (batch, 10)
          - score_cumulative: (batch, 13)
          - act_history: (batch, 16, _NUM_FUNCTIONS)
          - memory_state, carry_state: arbitrary (passed through)
        """
        super(FullyConv, self).__init__()
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.network_scale = screen_size // 32  # as in the original
        print("==FullyConv", screen_size, minimap_size)

        # feature_screen is assumed to have 24 channels.
        # self.screen_encoder = SpatialEncoder(height=screen_size, width=screen_size, channel=32)

        self.screen_encoder = nn.Sequential(
            nn.Conv2d(39, 24, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, padding=2),  # same padding when kernel=5 → pad=2
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, padding=1),  # kernel=3 → pad=1
            nn.ReLU(),
        )

        # After screen_encoder output (48 channels) we will concatenate the tiled
        # single_select and multi_select (each 32 channels) to get 112 channels.
        self.screen_input_encoder = nn.Sequential(
            nn.Conv2d(112, 39, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        # Dense "encoders" for the nonspatial features.
        self.single_select_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
        )
        self.multi_select_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
        )

        # Fully-connected layers for the flattened (concatenated) features.
        self.flattened_size = 112 * screen_size * screen_size
        self.feature_fc_1 = nn.Sequential(nn.Linear(self.flattened_size, 800), nn.ReLU())
        self.feature_fc_2 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_3 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_4 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_5 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())

        self.fn_out = nn.Linear(800, _NUM_FUNCTIONS)
        self.dense2 = nn.Linear(800, 1)

        # Output modules for spatial actions.
        self.screen = nn.Sequential(
            nn.Conv2d(39, 1, kernel_size=1, padding=0)
        )
        self.minimap = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, padding=0)
        )
        self.screen2 = nn.Sequential(
            nn.Conv2d(39, 1, kernel_size=1, padding=0)
        )
        # Output dense layers for nonspatial "heads."
        self.queued = nn.Linear(800, 2)
        self.control_group_act = nn.Linear(800, 5)
        self.control_group_id = nn.Linear(800, 10)
        self.select_point_act = nn.Linear(800, 4)
        self.select_add = nn.Linear(800, 2)
        self.select_unit_act = nn.Linear(800, 4)
        self.select_unit_id = nn.Linear(800, 500)
        self.select_worker = nn.Linear(800, 4)
        self.build_queue_id = nn.Linear(800, 10)
        self.unload_id = nn.Linear(800, 500)

    def forward(self, feature_screen, feature_minimap, player, feature_units, game_loop, available_actions,
                build_queue, single_select, multi_select, score_cumulative, act_history, memory_state, carry_state):
        if debug :
            print("feature_screen:", feature_screen.shape) 
            print("single_select:", single_select.shape) 
            print("multi_select:", multi_select.shape)
            print("feature_minimap:", feature_minimap.shape) 

        if feature_screen.dim() == 5:
            feature_screen = feature_screen.squeeze(0)
        if feature_screen.dim() == 4 and feature_screen.shape[1] != 39:
            # channel first (1, H, W, 39) -> (1, 39, H, W)
            feature_screen = feature_screen.permute(0, 3, 1, 2)

        if feature_minimap.dim() == 5:
            feature_minimap = feature_minimap.squeeze(0)
        if feature_minimap.dim() == 4 and feature_minimap.shape[1] != 27:
            feature_minimap = feature_minimap.permute(0, 3, 1, 2)

        if single_select.dim() == 3:
            # For example, if single_select has shape (1, batch, 3), remove the extra dimension.
            single_select = single_select.squeeze(0)
        if multi_select.dim() == 3:
            multi_select = multi_select.squeeze(0)

        if debug :
            print("feature_screen:", feature_screen.shape) 
            print("single_select:", single_select.shape) 
            print("multi_select:", multi_select.shape)
            print("feature_minimap:", feature_minimap.shape) 

        batch_size, _, H, W = feature_screen.size()

        # --- Convolutional (spatial) processing of the screen input ---
        feature_screen_encoded = self.screen_encoder(feature_screen)  # -> (batch, 48, H, W)

        # Process nonspatial single and multi select inputs.
        # single_select: (batch, 3) --> (batch, 32) then expand to (batch, 32, H, W)
        single_select_encoded = self.single_select_encoder(single_select)
        single_select_encoded = single_select_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, 32, H, W)

        multi_select_encoded = self.multi_select_encoder(multi_select)
        multi_select_encoded = multi_select_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, 32, H, W)

        # Concatenate along the channel dimension
        feature_encoded = torch.cat([feature_screen_encoded, single_select_encoded, multi_select_encoded], dim=1)  # (batch, 112, H, W)

        feature_encoded_for_screen = self.screen_input_encoder(feature_encoded)  # (batch, 24, H, W)

        # Residual addition (note that feature_screen is the original input with 24 channels)
        print() if debug else None
        print("feature_encoded_for_screen:", feature_encoded_for_screen.shape) if debug else None
        print("feature_screen:", feature_screen.shape) if debug else None
        screen_input = F.relu(feature_encoded_for_screen + feature_screen)

        # --- Fully connected (nonspatial) branch ---
        # Flatten the concatenated features.
        feature_encoded_flat = feature_encoded.view(batch_size, -1)  # (batch, 112*H*W)
        feature_fc = self.feature_fc_1(feature_encoded_flat)
        feature_fc = self.feature_fc_2(feature_fc)
        feature_fc = self.feature_fc_3(feature_fc)
        feature_fc = self.feature_fc_4(feature_fc)
        feature_fc = self.feature_fc_5(feature_fc)

        fn_out = self.fn_out(feature_fc)
        value = self.dense2(feature_fc)

        final_memory_state = memory_state
        final_carry_state = carry_state

        # --- Additional spatial outputs ---
        screen_args = self.screen(screen_input)  # (batch, 1, H, W)
        screen_args = screen_args.view(batch_size, -1)

        minimap_args = self.minimap(feature_minimap)  # (batch, 1, minimap_H, minimap_W)
        minimap_args = minimap_args.view(batch_size, -1)

        screen2_args = self.screen2(screen_input)
        screen2_args = screen2_args.view(batch_size, -1)

        # --- Nonspatial heads ---
        queued_args = self.queued(feature_fc)
        control_group_act_args = self.control_group_act(feature_fc)
        control_group_id_args = self.control_group_id(feature_fc)
        select_point_act_args = self.select_point_act(feature_fc)
        select_add_args = self.select_add(feature_fc)
        select_unit_act_args = self.select_unit_act(feature_fc)
        select_unit_id_args = self.select_unit_id(feature_fc)
        select_worker_args = self.select_worker(feature_fc)
        build_queue_id_args = self.build_queue_id(feature_fc)
        unload_id_args = self.unload_id(feature_fc)

        return (fn_out,
                [screen_args, minimap_args, screen2_args, queued_args,
                control_group_act_args, control_group_id_args,
                select_point_act_args, select_add_args, select_unit_act_args, select_unit_id_args,
                select_worker_args, build_queue_id_args, unload_id_args],
                value, final_memory_state, final_carry_state)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Multi-head attention module.
        d_model: dimensionality of input and output.
        num_heads: number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Note: In the original Keras code, the layer normalization is applied outside.
        # Here we leave that to the caller.
    
    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, d_model)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)  # -> (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention.
        dk = self.depth
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (batch, num_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, depth)

        # Concatenate heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(attn_output)
        # In the original code the same dense is applied twice.
        output = self.dense(output)
        output = self.dropout(output)
        return output, attn_weights

class RelationalFullyConv(nn.Module):
    def __init__(self, screen_size, minimap_size):
        super(RelationalFullyConv, self).__init__()
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.network_scale = screen_size // 32
        self._conv_out_size_screen = screen_size // 2  # Since stride=2 in screen_encoder

        # Location embeddings
        self._locs_screen = torch.arange(0, self._conv_out_size_screen**2, dtype=torch.float32) / (self._conv_out_size_screen**2)
        self._locs_screen = self._locs_screen.view(1, -1, 1)

        self.screen_encoder = nn.Sequential(
            nn.Conv2d(39, 47, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.attention_screen_1 = MultiHeadAttention(48, 4)
        self.layernorm_screen_1 = nn.LayerNorm(48)
        self.dropout_screen_1 = nn.Dropout(0.1)

        self.attention_screen_2 = MultiHeadAttention(48, 4)
        self.layernorm_screen_2 = nn.LayerNorm(48)
        self.dropout_screen_2 = nn.Dropout(0.1)

        self.screen_decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.single_select_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU()
        )

        self.act_history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * _NUM_FUNCTIONS, 64),
            nn.ReLU()
        )

        # Feature encoder to match feature_screen channels for residual addition
        self.feature_encoder = nn.Conv2d(48 + 16 + 64, 39, kernel_size=1)

        # Fully connected layers for flattened features
        self.feature_fc = nn.Sequential(
            nn.Linear((48 + 16 + 64) * screen_size * screen_size, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU()
        )

        self.fn_out = nn.Linear(800, _NUM_FUNCTIONS)
        self.dense2 = nn.Linear(800, 1)

        # Output modules for spatial actions
        self.screen = nn.Sequential(
            nn.Conv2d(39, 1, kernel_size=1),
            nn.Flatten()
        )
        self.minimap = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Flatten()
        )
        self.screen2 = nn.Sequential(
            nn.Conv2d(39, 1, kernel_size=1),
            nn.Flatten()
        )

        # Output dense layers for nonspatial "heads"
        self.queued = nn.Linear(800, 2)
        self.control_group_act = nn.Linear(800, 5)
        self.control_group_id = nn.Linear(800, 10)
        self.select_point_act = nn.Linear(800, 4)
        self.select_add = nn.Linear(800, 2)
        self.select_unit_act = nn.Linear(800, 4)
        self.select_unit_id = nn.Linear(800, 500)
        self.select_worker = nn.Linear(800, 4)
        self.build_queue_id = nn.Linear(800, 10)
        self.unload_id = nn.Linear(800, 500)

    def forward(self, feature_screen, feature_minimap, player, feature_units, game_loop, available_actions,
                build_queue, single_select, multi_select, score_cumulative, act_history, memory_state, carry_state):
        
        if feature_screen.dim() == 5:
            feature_screen = feature_screen.squeeze(0)
        if feature_screen.dim() == 4 and feature_screen.shape[1] != 39:
            # channel first (1, H, W, 39) -> (1, 39, H, W)
            feature_screen = feature_screen.permute(0, 3, 1, 2)

        if feature_minimap.dim() == 5:
            feature_minimap = feature_minimap.squeeze(0)
        if feature_minimap.dim() == 4 and feature_minimap.shape[1] != 27:
            feature_minimap = feature_minimap.permute(0, 3, 1, 2)

        if single_select.dim() == 3:
            # For example, if single_select has shape (1, batch, 3), remove the extra dimension.
            single_select = single_select.squeeze(0)
        if multi_select.dim() == 3:
            multi_select = multi_select.squeeze(0)
        
        batch_size, _, H, W = feature_screen.size()
        # batch_size = feature_screen.size(0)

        # Fix: Ensure act_history's batch dimension matches feature_screen's batch dimension
        if act_history.dim() == 3 and act_history.size(0) == 1 and batch_size != 1:
            act_history = act_history.squeeze(0)

        # Preprocess act_history to ensure it is one-hot encoded and has shape 
        # [batch_size, seq_length, _NUM_FUNCTIONS]
        if act_history.dim() == 2 and act_history.size(-1) != _NUM_FUNCTIONS:
            act_history = torch.nn.functional.one_hot(act_history.long(), num_classes=_NUM_FUNCTIONS).float()
        elif act_history.dim() == 3 and act_history.size(-1) != _NUM_FUNCTIONS:
            act_history = torch.nn.functional.one_hot(act_history.squeeze(-1).long(), num_classes=_NUM_FUNCTIONS).float()
        
        # Ensure act_history has the correct shape: [batch_size, 16, _NUM_FUNCTIONS]
        if act_history.dim() == 2:
            act_history = act_history.unsqueeze(1)
        if act_history.size(1) != 16:
            if act_history.size(1) < 16:
                padding = torch.zeros(batch_size, 16 - act_history.size(1), _NUM_FUNCTIONS, 
                                        device=act_history.device)
                act_history = torch.cat([act_history, padding], dim=1)
            else:
                act_history = act_history[:, :16, :]

        # Screen encoding
        feature_screen_encoded = self.screen_encoder(feature_screen)  # Shape: (batch, 47, H/2, W/2)
        feature_screen_encoded_attention = feature_screen_encoded.permute(0, 2, 3, 1).view(batch_size, -1, 47)  # Shape: (batch, (H/2)*(W/2), 47)

        num_locations = feature_screen_encoded_attention.size(1)
        self._locs_screen = torch.arange(0, num_locations, dtype=torch.float32, device=feature_screen.device) / num_locations
        self._locs_screen = self._locs_screen.view(1, -1, 1).expand(batch_size, -1, 1)  # Shape: (batch, (H/2)*(W/2), 1)
        feature_screen_encoded_locs = torch.cat([feature_screen_encoded_attention, self._locs_screen], dim=2)  # Shape: (batch, (H/2)*(W/2), 48)


        locs_screen = self._locs_screen.expand(batch_size, -1, 1)  # Shape: (batch, (H/2)*(W/2), 1)
        # feature_screen_encoded_locs = torch.cat([feature_screen_encoded_attention, locs_screen], dim=2)  # Shape: (batch, (H/2)*(W/2), 48)

        # First attention block
        attention_feature_screen_1, _ = self.attention_screen_1(feature_screen_encoded_locs,
                                                               feature_screen_encoded_locs,
                                                               feature_screen_encoded_locs)
        attention_feature_screen_1 = self.dropout_screen_1(attention_feature_screen_1)
        attention_feature_screen_1 = self.layernorm_screen_1(feature_screen_encoded_locs + attention_feature_screen_1)

        # Second attention block
        attention_feature_screen_2, _ = self.attention_screen_2(attention_feature_screen_1,
                                                               attention_feature_screen_1,
                                                               attention_feature_screen_1)
        attention_feature_screen_2 = self.dropout_screen_2(attention_feature_screen_2)
        attention_feature_screen_2 = self.layernorm_screen_2(attention_feature_screen_1 + attention_feature_screen_2)

        # Reshape and decode
        # Calculate the actual spatial dimensions based on the input size
        spatial_dim = int(math.sqrt(attention_feature_screen_2.size(1)))
        relational_spatial = attention_feature_screen_2.view(batch_size, spatial_dim, spatial_dim, 48).permute(0, 3, 1, 2)  # Shape: (batch, 48, H/2, W/2)
        relational_spatial = self.screen_decoder(relational_spatial)  # Shape: (batch, 48, H, W)

        # Non-spatial encodings
        single_select_encoded = self.single_select_encoder(single_select)  # Shape: (batch, 16)
        act_history = act_history.float()
        act_history_encoded = self.act_history_encoder(act_history)  # Shape: (batch, 64)

        # Get the spatial dimensions from relational_spatial output
        H_dec, W_dec = relational_spatial.size(2), relational_spatial.size(3)

        single_select_encoded = single_select_encoded.view(batch_size, 16, 1, 1).expand(-1, 16, H_dec, W_dec)  # Shape: (batch, 16, H_dec, W_dec)
        act_history_encoded = act_history_encoded.view(batch_size, 64, 1, 1).expand(-1, 64, H_dec, W_dec)  # Shape: (batch, 64, H_dec, W_dec)

        # Concatenate spatial and tiled non-spatial features.
        feature_spatial = torch.cat([relational_spatial, single_select_encoded, act_history_encoded], dim=1)  # Shape: (batch, 128, H_dec, W_dec)
        feature_spatial_encoded = self.feature_encoder(feature_spatial)  # Shape: (batch, 39, H, W)

        # Residual addition with original feature_screen
        screen_input = F.relu(feature_spatial_encoded + feature_screen)  # Shape: (batch, 39, H, W)

        # Fully connected branch
        feature_spatial_flatten = feature_spatial.view(batch_size, -1)  # Shape: (batch, 128*H*W)
        feature_fc = self.feature_fc(feature_spatial_flatten)  # Shape: (batch, 800)

        # Outputs
        fn_out = self.fn_out(feature_fc)  # Shape: (batch, _NUM_FUNCTIONS)
        value = self.dense2(feature_fc)  # Shape: (batch, 1)

        # Spatial outputs with flattening
        screen_args_out = self.screen(screen_input)  # Shape: (batch, H*W)
        minimap_args_out = self.minimap(feature_minimap)  # Shape: (batch, minimap_H*minimap_W)
        screen2_args_out = self.screen2(screen_input)  # Shape: (batch, H*W)

        # Non-spatial outputs
        queued_args_out = self.queued(feature_fc)
        control_group_act_args_out = self.control_group_act(feature_fc)
        control_group_id_args_out = self.control_group_id(feature_fc)
        select_point_act_args_out = self.select_point_act(feature_fc)
        select_add_args_out = self.select_add(feature_fc)
        select_unit_act_args_out = self.select_unit_act(feature_fc)
        select_unit_id_args_out = self.select_unit_id(feature_fc)
        select_worker_args_out = self.select_worker(feature_fc)
        build_queue_id_args_out = self.build_queue_id(feature_fc)
        unload_id_args_out = self.unload_id(feature_fc)

        final_memory_state = memory_state
        final_carry_state = carry_state

        return (fn_out,
                [screen_args_out, minimap_args_out, screen2_args_out, queued_args_out,
                 control_group_act_args_out, control_group_id_args_out,
                 select_point_act_args_out, select_add_args_out, select_unit_act_args_out, select_unit_id_args_out,
                 select_worker_args_out, build_queue_id_args_out, unload_id_args_out],
                value, final_memory_state, final_carry_state)

def make_model(name):
    """
    Returns an instance of the model based on the name.
    Choose "fullyconv" or "relationalfullyconv".
    """
    print("name:", name)
    if name == 'fullyconv':
        return FullyConv(screen_size=32, minimap_size=32)
    elif name == 'relationalfullyconv':
        return RelationalFullyConv(screen_size=32, minimap_size=32)
    else:
        raise ValueError("Unknown model name: {}".format(name))