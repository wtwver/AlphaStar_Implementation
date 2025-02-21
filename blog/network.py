import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pysc2.lib import actions
# _NUM_FUNCTIONS is defined based on the available actions.
_NUM_FUNCTIONS = len(actions.FUNCTIONS)


class FullyConv(nn.Module):
    def __init__(self, screen_size, minimap_size):
        """
        Fully convolutional network version.

        Expected inputs (in PyTorch “channels-first” format):
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
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, padding=2),  # same padding when kernel=5 → pad=2
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, padding=1),  # kernel=3 → pad=1
            nn.ReLU(),
        )

        # After screen_encoder output (48 channels) we will concatenate the tiled
        # single_select and multi_select (each 32 channels) to get 112 channels.
        self.screen_input_encoder = nn.Sequential(
            nn.Conv2d(112, 24, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        # Dense “encoders” for the nonspatial features.
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
            nn.Conv2d(24, 1, kernel_size=1, padding=0)
        )
        self.minimap = nn.Sequential(
            nn.Conv2d(7, 1, kernel_size=1, padding=0)
        )
        self.screen2 = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0)
        )
        # Output dense layers for nonspatial “heads.”
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
        print(feature_screen)
        # feature_screen: (batch, 24, H, W)
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
                screen_args, minimap_args, screen2_args, queued_args,
                control_group_act_args, control_group_id_args,
                select_point_act_args, select_add_args, select_unit_act_args, select_unit_id_args,
                select_worker_args, build_queue_id_args, unload_id_args,
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
        """
        Relational fully convolutional network version.
        Expected input dimensions are similar to FullyConv (channels-first).
        """
        super(RelationalFullyConv, self).__init__()
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.network_scale = screen_size // 32

        # screen_encoder: Conv2d with stride 2, roughly halving spatial dims.
        self.screen_encoder = nn.Sequential(
            nn.Conv2d(24, 47, kernel_size=3, stride=2, padding=1),  # output: (batch, 47, 8, 8) if input is 16x16.
            nn.ReLU(),
        )

        # Two blocks of multi-head attention.
        # We will add one additional channel (the "locs") so that d_model=47+1=48.
        self.attention_screen_1 = MultiHeadAttention(d_model=48, num_heads=4, dropout=0.1)
        self.dropout_screen_1 = nn.Dropout(0.1)
        self.layernorm_screen_1 = nn.LayerNorm(48)
        
        self.attention_screen_2 = MultiHeadAttention(d_model=48, num_heads=4, dropout=0.1)
        self.dropout_screen_2 = nn.Dropout(0.1)
        self.layernorm_screen_2 = nn.LayerNorm(48)

        # The output spatial size after screen_encoder is 8 (i.e. 8x8)
        self._conv_out_size_screen = 8
        # Build a buffer for location encoding.
        locs = [i / (self._conv_out_size_screen * self._conv_out_size_screen)
                for i in range(self._conv_out_size_screen * self._conv_out_size_screen)]
        locs = torch.tensor(locs, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # shape: (1, 64, 1)
        self.register_buffer("locs_screen", locs)

        # A 1x1 convolution to combine the relational features.
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(128, 24, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.single_select_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
        )
        # act_history encoder: flatten then linear to 64.
        self.act_history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * _NUM_FUNCTIONS, 64),
            nn.ReLU(),
        )

        # screen_decoder: Transposed convolution to go from 8x8 to screen_size (16x16).
        self.screen_decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )

        # For the fully-connected branch.
        self.flattened_size_rel = 128 * screen_size * screen_size  # 128 channels after concatenation.
        self.feature_fc_1 = nn.Sequential(nn.Linear(self.flattened_size_rel, 800), nn.ReLU())
        self.feature_fc_2 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_3 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_4 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())
        self.feature_fc_5 = nn.Sequential(nn.Linear(800, 800), nn.ReLU())

        self.fn_out = nn.Linear(800, _NUM_FUNCTIONS)
        self.dense2 = nn.Linear(800, 1)

        # Output modules (the same as in FullyConv).
        self.screen = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0)
        )
        self.minimap = nn.Sequential(
            nn.Conv2d(7, 1, kernel_size=1, padding=0)
        )
        self.screen2 = nn.Sequential(
            nn.Conv2d(24, 1, kernel_size=1, padding=0)
        )
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
        batch_size = feature_screen.size(0)

        # --- Relational spatial processing ---
        # feature_screen: (batch, 24, 16, 16)
        # Pass through screen_encoder: -> (batch, 47, 8, 8)
        x = self.screen_encoder(feature_screen)  # (batch, 47, 8, 8)
        # Flatten spatial dims: (batch, 64, 47)
        x = x.view(batch_size, self._conv_out_size_screen * self._conv_out_size_screen, 47)

        # Tile/add location encoding.
        locs = self.locs_screen.expand(batch_size, -1, -1)  # (batch, 64, 1)
        x = torch.cat([x, locs], dim=2)  # (batch, 64, 48)

        # First attention block.
        attn1, _ = self.attention_screen_1(x, x, x, mask=None)
        attn1 = self.dropout_screen_1(attn1)
        # Apply layernorm over last dimension.
        attn1 = self.layernorm_screen_1(x + attn1)

        # Second attention block.
        attn2, _ = self.attention_screen_2(attn1, attn1, attn1, mask=None)
        attn2 = self.dropout_screen_2(attn2)
        attn2 = self.layernorm_screen_2(x + attn2)

        # Reshape back to spatial map.
        # attn2: (batch, 64, 48) -> reshape to (batch, 8, 8, 48) then permute to (batch, 48, 8, 8)
        relational_spatial = attn2.view(batch_size, self._conv_out_size_screen, self._conv_out_size_screen, 48)
        relational_spatial = relational_spatial.permute(0, 3, 1, 2)  # (batch, 48, 8, 8)

        # Decode: upsample with transposed conv.
        relational_spatial = self.screen_decoder(relational_spatial)  # (batch, 48, screen_size, screen_size)

        # --- Process nonspatial single_select and act_history ---
        single_select_encoded = self.single_select_encoder(single_select)  # (batch, 16)
        single_select_encoded = single_select_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, 16, self.screen_size, self.screen_size)

        act_history_encoded = self.act_history_encoder(act_history)  # (batch, 64)
        act_history_encoded = act_history_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, 64, self.screen_size, self.screen_size)

        # Concatenate along channel dimension.
        # relational_spatial: (batch, 48, screen_size, screen_size)
        # single_select_encoded: (batch, 16, screen_size, screen_size)
        # act_history_encoded: (batch, 64, screen_size, screen_size)
        feature_spatial = torch.cat([relational_spatial, single_select_encoded, act_history_encoded], dim=1)  # (batch, 128, screen_size, screen_size)

        # Pass through 1x1 conv encoder.
        feature_spatial_encoded = self.feature_encoder(feature_spatial)  # (batch, 24, screen_size, screen_size)

        # Residual addition with the raw feature_screen.
        screen_input = F.relu(feature_spatial_encoded + feature_screen)  # (batch, 24, screen_size, screen_size)
        minimap_input = feature_minimap

        # --- Fully-connected branch ---
        # Flatten the spatial feature (from the concatenated features, not the one after 1x1 conv).
        feature_spatial_flat = feature_spatial.view(batch_size, -1)  # (batch, 128*screen_size*screen_size)
        feature_fc = self.feature_fc_1(feature_spatial_flat)
        feature_fc = self.feature_fc_2(feature_fc)
        feature_fc = self.feature_fc_3(feature_fc)
        feature_fc = self.feature_fc_4(feature_fc)
        feature_fc = self.feature_fc_5(feature_fc)

        fn_out = self.fn_out(feature_fc)
        value = self.dense2(feature_fc)

        final_memory_state = memory_state
        final_carry_state = carry_state

        # --- Additional spatial outputs ---
        screen_args = self.screen(screen_input)
        screen_args = screen_args.view(batch_size, -1)

        minimap_args = self.minimap(minimap_input)
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
                screen_args, minimap_args, screen2_args, queued_args,
                control_group_act_args, control_group_id_args,
                select_point_act_args, select_add_args, select_unit_act_args, select_unit_id_args,
                select_worker_args, build_queue_id_args, unload_id_args,
                value, final_memory_state, final_carry_state)


def make_model(name):
    """
    Returns an instance of the model based on the name.
    Choose "fullyconv" or "relationalfullyconv".
    """
    if name == 'fullyconv':
        return FullyConv(screen_size=16, minimap_size=16)
    elif name == 'relationalfullyconv':
        return RelationalFullyConv(screen_size=16, minimap_size=16)
    else:
        raise ValueError("Unknown model name: {}".format(name))