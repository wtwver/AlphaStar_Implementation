class AlphaStar(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(AlphaStar, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)

    # State Encoder
    self.screen_encoder = SpatialEncoder(height=screen_size, width=screen_size, channel=32)
    self.minimap_encoder = SpatialEncoder(height=minimap_size, width=minimap_size, channel=16)
    self.player_encoder = ScalarEncoder(output_dim=13)
    self.game_loop_encoder = ScalarEncoder(output_dim=64)
    #self.entity_encoder = EntityEncoder(output_dim=11, entity_num=50)
    self.available_actions_encoder = ScalarEncoder(output_dim=64)
    self.build_queue_encoder = ScalarEncoder(output_dim=5)
    self.single_select_encoder = ScalarEncoder(output_dim=5)
    self.multi_select_encoder = ScalarEncoder(output_dim=10)
    self.score_cumulative_encoder = ScalarEncoder(output_dim=10)

    self.encoding_lookup = utils.positional_encoding(max_position=20000, embedding_size=64)

    # Core
    self.core = Core(256, self.network_scale)

    # Action Head
    self.action_type_head = ActionTypeHead(_NUM_FUNCTIONS, self.network_scale)
    self.screen_argument_head = SpatialArgumentHead(height=screen_size, width=screen_size)
    self.minimap_argument_head = SpatialArgumentHead(height=minimap_size, width=minimap_size)
    self.screen2_argument_head = SpatialArgumentHead(height=screen_size, width=screen_size)
    self.queued_argument_head = ScalarArgumentHead(2)
    self.control_group_act_argument_head = ScalarArgumentHead(5)
    self.control_group_id_argument_head = ScalarArgumentHead(10)
    self.select_point_act_argument_head = ScalarArgumentHead(4)
    self.select_add_argument_head = ScalarArgumentHead(2)
    self.select_unit_act_argument_head = ScalarArgumentHead(4)
    self.select_unit_id_argument_head = ScalarArgumentHead(500)
    self.select_worker_argument_head = ScalarArgumentHead(4)
    self.build_queue_id_argument_head = ScalarArgumentHead(10)
    self.unload_id_argument_head = ScalarArgumentHead(50)

    self.baseline = Baseline(256)
    self.args_out_logits = dict()

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out_logits': self.args_out_logits
    })
    return config

  def call(self, feature_screen, feature_minimap, player, feature_units, memory_state, carry_state, game_loop,
             available_actions, build_queue, single_select, multi_select, score_cumulative):
    batch_size = tf.shape(feature_screen)[0]

    feature_screen_encoded = self.screen_encoder(feature_screen)

    feature_minimap_encoded = self.minimap_encoder(feature_minimap)

    player_encoded = self.player_encoder(player)
    player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(player_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    player_encoded = tf.cast(player_encoded, 'float32')
    
    game_loop_encoded = tf.gather_nd(self.encoding_lookup, tf.cast(game_loop, tf.int32))
    game_loop_encoded = self.game_loop_encoder(game_loop_encoded)
    game_loop_encoded = tf.tile(tf.expand_dims(tf.expand_dims(game_loop_encoded, 1), 2),
                                      tf.stack([1, self.screen_size, self.screen_size, 1]))
    game_loop_encoded = tf.cast(game_loop_encoded, 'float32')

    #feature_units_encoded = self.entity_encoder(feature_units)
    #feature_units_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_units_encoded, 1), 2),
    #                                    tf.stack([1, self.screen_size, self.screen_size, 1]))
    #feature_units_encoded = tf.cast(feature_units_encoded, 'float32')

    available_actions_encoded = self.available_actions_encoder(available_actions)
    available_actions_encoded = tf.tile(tf.expand_dims(tf.expand_dims(available_actions_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    available_actions_encoded = tf.cast(available_actions_encoded, 'float32')

    build_queue_encoded = self.build_queue_encoder(build_queue)
    build_queue_encoded = tf.tile(tf.expand_dims(tf.expand_dims(build_queue_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    build_queue_encoded = tf.cast(build_queue_encoded, 'float32')

    single_select_encoded = self.single_select_encoder(single_select)
    single_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(single_select_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    single_select_encoded = tf.cast(single_select_encoded, 'float32')

    multi_select_encoded = self.multi_select_encoder(multi_select)
    multi_select_encoded = tf.tile(tf.expand_dims(tf.expand_dims(multi_select_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    multi_select_encoded = tf.cast(multi_select_encoded, 'float32')

    score_cumulative_encoded = self.score_cumulative_encoder(score_cumulative)
    score_cumulative_encoded = tf.tile(tf.expand_dims(tf.expand_dims(score_cumulative_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    score_cumulative_encoded = tf.cast(score_cumulative_encoded, 'float32')
    
    feature_encoded = tf.concat([feature_screen_encoded, feature_minimap_encoded, player_encoded, game_loop_encoded, 
                                       available_actions_encoded, build_queue_encoded, single_select_encoded, multi_select_encoded,
                                       score_cumulative_encoded], axis=3)

    core_outputs, memory_state, carry_state = self.core(feature_encoded, memory_state, carry_state)

    action_type_logits, autoregressive_embedding = self.action_type_head(core_outputs)
    
    for arg_type in actions.TYPES:
      if arg_type.name == 'screen':
        self.args_out_logits[arg_type] = self.screen_argument_head(feature_screen_encoded, core_outputs, autoregressive_embedding)
      elif arg_type.name == 'minimap':
        self.args_out_logits[arg_type] = self.minimap_argument_head(feature_minimap_encoded, core_outputs, autoregressive_embedding)
      elif arg_type.name == 'screen2':
        self.args_out_logits[arg_type] = self.screen2_argument_head(feature_screen_encoded, core_outputs, autoregressive_embedding)
      elif arg_type.name == 'queued':
        self.args_out_logits[arg_type] = self.queued_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'control_group_act':
        self.args_out_logits[arg_type] = self.control_group_act_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'control_group_id':
        self.args_out_logits[arg_type] = self.control_group_id_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'select_point_act':
        self.args_out_logits[arg_type] = self.select_point_act_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'select_add':
        self.args_out_logits[arg_type] = self.select_add_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'select_unit_act':
        self.args_out_logits[arg_type] = self.select_unit_act_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'select_unit_id':
        self.args_out_logits[arg_type] = self.select_unit_id_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'select_worker':
        self.args_out_logits[arg_type] = self.select_worker_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'build_queue_id':
        self.args_out_logits[arg_type] = self.build_queue_id_argument_head(core_outputs, autoregressive_embedding)
      elif arg_type.name == 'unload_id':
        self.args_out_logits[arg_type] = self.unload_id_argument_head(core_outputs, autoregressive_embedding)

    value = self.baseline(core_outputs)

    return action_type_logits, self.args_out_logits, value, memory_state, carry_state
