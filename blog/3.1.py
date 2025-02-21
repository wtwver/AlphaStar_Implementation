import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True) 

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 

  output = tf.matmul(attention_weights, v)  

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wk = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')
    self.wv = tf.keras.layers.Dense(d_model, kernel_regularizer='l2')

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)
    self.dense = tf.keras.layers.Dense(d_model)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
    })
    return config
    
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training):
    batch_size = tf.shape(q)[0]
    
    v_original = v
    
    q = self.wq(q)  
    k = self.wk(k) 
    v = self.wv(v)  

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)  
    v = self.split_heads(v, batch_size) 

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])  
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) 

    output = self.dense(concat_attention)
    output = self.dense(output) 
    
    return output, attention_weights


class EntityEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim, entity_num):
    super(EntityEncoder, self).__init__()
    self.output_dim = output_dim

    self.attention = MultiHeadAttention(8, 1)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout = tf.keras.layers.Dropout(0.1)

    self.entity_num = entity_num
    self.locs = []
    for i in range(0, self.entity_num):
        self.locs.append(i / float(self.entity_num))
            
    self.locs = tf.expand_dims(self.locs, 0)
    self.locs = tf.expand_dims(self.locs, 2)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim,
        'entity_num': self.entity_num
    })
    
    return config

  def call(self, entity_features, training):
    batch_size = tf.shape(entity_features)[0]

    locs = tf.tile(self.locs, [batch_size, 1, 1])
    entity_features_locs = tf.concat([entity_features, locs], 2)
    attention_output, _ = self.attention(entity_features_locs, entity_features_locs, entity_features_locs, None)
        
    attention_output = self.dropout(attention_output, training=training)
    attention_output = self.layernorm(entity_features_locs + attention_output)
    max_pool_1d = tf.math.reduce_max(attention_output, 1)
    output = max_pool_1d

    return output


class FullyConv(tf.keras.Model):
  def __init__(self, screen_size, minimap_size):
    super(FullyConv, self).__init__()

    self.screen_size = screen_size
    self.minimap_size = minimap_size

    self.network_scale = int(screen_size / 32)
    
    self.screen_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
    ])
    
    self.minimap_encoder = tf.keras.Sequential([
       tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(8, 1, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu'),
       tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
    ])
    
    self.player_encoder = tf.keras.layers.Dense(11, activation='relu')
    
    self.entity_encoder = EntityEncoder(output_dim=11, entity_num=3)
    
    self.available_actions_encoder = tf.keras.layers.Dense(128, activation='relu')
    
    self.feature_fc = tf.keras.layers.Dense(512, activation='relu')
    self.fn_out = tf.keras.layers.Dense(_NUM_FUNCTIONS, activation='softmax')

    self.screen = tf.keras.Sequential()
    self.screen.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen.add(tf.keras.layers.Flatten())
    self.screen.add(tf.keras.layers.Softmax())

    self.minimap = tf.keras.Sequential()
    self.minimap.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.minimap.add(tf.keras.layers.Flatten())
    self.minimap.add(tf.keras.layers.Softmax())

    self.screen2 = tf.keras.Sequential()
    self.screen2.add(tf.keras.layers.Conv2D(1, 1, padding='same'))
    self.screen2.add(tf.keras.layers.Flatten())
    self.screen2.add(tf.keras.layers.Softmax())

    self.queued = tf.keras.Sequential()
    self.queued.add(tf.keras.layers.Dense(2))
    self.queued.add(tf.keras.layers.Softmax())

    self.control_group_act = tf.keras.Sequential()
    self.control_group_act.add(tf.keras.layers.Dense(5))
    self.control_group_act.add(tf.keras.layers.Softmax())

    self.control_group_id = tf.keras.Sequential()
    self.control_group_id.add(tf.keras.layers.Dense(10))
    self.control_group_id.add(tf.keras.layers.Softmax())

    self.select_point_act = tf.keras.Sequential()
    self.select_point_act.add(tf.keras.layers.Dense(4))
    self.select_point_act.add(tf.keras.layers.Softmax())

    self.select_add = tf.keras.Sequential()
    self.select_add.add(tf.keras.layers.Dense(2))
    self.select_add.add(tf.keras.layers.Softmax())

    self.select_unit_act = tf.keras.Sequential()
    self.select_unit_act.add(tf.keras.layers.Dense(4))
    self.select_unit_act.add(tf.keras.layers.Softmax())

    self.select_unit_id = tf.keras.Sequential()
    self.select_unit_id.add(tf.keras.layers.Dense(500))
    self.select_unit_id.add(tf.keras.layers.Softmax())

    self.select_worker = tf.keras.Sequential()
    self.select_worker.add(tf.keras.layers.Dense(4))
    self.select_worker.add(tf.keras.layers.Softmax())

    self.build_queue_id = tf.keras.Sequential()
    self.build_queue_id.add(tf.keras.layers.Dense(10))
    self.build_queue_id.add(tf.keras.layers.Softmax())

    self.unload_id = tf.keras.Sequential()
    self.unload_id.add(tf.keras.layers.Dense(500))
    self.unload_id.add(tf.keras.layers.Softmax())

    self.dense2 = tf.keras.layers.Dense(1)
   
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'args_out': self.args_out,
        'input_shape': self.input_shape
    })
    return config

  def call(self, feature_screen, feature_minimap, player, feature_units, memory_state, carry_state, game_loop, available_actions, 
  	   build_queue, single_select, multi_select, score_cumulative):
    batch_size = tf.shape(feature_screen)[0]

    feature_screen_encoded = self.screen_encoder(feature_screen)

    feature_minimap_encoded = self.minimap_encoder(feature_minimap)

    player_encoded = self.player_encoder(player)
    player_encoded = tf.tile(tf.expand_dims(tf.expand_dims(player_encoded, 1), 2),
                                     tf.stack([1, 32, 32, 1]))
    player_encoded = tf.cast(player_encoded, 'float32')
    
    feature_units_encoded = self.entity_encoder(feature_units)
    feature_units_encoded = tf.tile(tf.expand_dims(tf.expand_dims(feature_units_encoded, 1), 2),
                                    tf.stack([1, self.screen_size, self.screen_size, 1]))
    feature_units_encoded = tf.cast(feature_units_encoded, 'float32')
    
    available_actions_encoded = self.available_actions_encoder(available_actions)
    available_actions_encoded = tf.tile(tf.expand_dims(tf.expand_dims(available_actions_encoded, 1), 2),
                                            tf.stack([1, self.screen_size, self.screen_size, 1]))
    available_actions_encoded = tf.cast(available_actions_encoded, 'float32')
    
    feature_encoded = tf.concat([feature_screen_encoded, feature_minimap_encoded, player_encoded, feature_units_encoded,
    				  available_actions_encoded], axis=3)

    feature_encoded_flatten = Flatten()(feature_encoded)
    feature_fc = self.feature_fc(feature_encoded_flatten)

    fn_out = self.fn_out(feature_fc)
    
    args_out = dict()
    for arg_type in actions.TYPES:
      if arg_type.name == 'screen':
        args_out[arg_type] = self.screen(feature_encoded)
      elif arg_type.name == 'minimap':
        args_out[arg_type] = self.minimap(feature_encoded)
      elif arg_type.name == 'screen2':
        args_out[arg_type] = self.screen2(feature_encoded)
      elif arg_type.name == 'queued':
        args_out[arg_type] = self.queued(feature_fc)
      elif arg_type.name == 'control_group_act':
        args_out[arg_type] = self.control_group_act(feature_fc)
      elif arg_type.name == 'control_group_id':
        args_out[arg_type] = self.control_group_id(feature_fc)
      elif arg_type.name == 'select_point_act':
        args_out[arg_type] = self.select_point_act(feature_fc)
      elif arg_type.name == 'select_add':
        args_out[arg_type] = self.select_add(feature_fc)
      elif arg_type.name == 'select_unit_act':
        args_out[arg_type] = self.select_unit_act(feature_fc)
      elif arg_type.name == 'select_unit_id':
        args_out[arg_type] = self.select_unit_id(feature_fc)
      elif arg_type.name == 'select_worker':
        args_out[arg_type] = self.select_worker(feature_fc)
      elif arg_type.name == 'build_queue_id':
        args_out[arg_type] = self.build_queue_id(feature_fc)
      elif arg_type.name == 'unload_id':
        args_out[arg_type] = self.unload_id(feature_fc)

    value = self.dense2(feature_fc)
    
    final_memory_state = memory_state
    final_carry_state = carry_state

    return fn_out, args_out, value, final_memory_state, final_carry_state