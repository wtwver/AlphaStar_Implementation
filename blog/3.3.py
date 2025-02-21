def run_episode(
    initial_feature_screen: tf.Tensor,
    initial_feature_minimap: tf.Tensor,
    initial_player: tf.Tensor,
    initial_feature_units: tf.Tensor,
    initial_game_loop: tf.Tensor,
    initial_available_actions: tf.Tensor,
    initial_build_queue: tf.Tensor,
    initial_single_select: tf.Tensor,
    initial_multi_select: tf.Tensor,
    initial_score_cumulative: tf.Tensor,
    initial_memory_state: tf.Tensor, 
    initial_carry_state: tf.Tensor, 
    initial_done: tf.Tensor, 
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                tf.Tensor]:
  """Runs a single episode to collect training data."""

  fn_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  screen_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  minimap_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  screen2_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  queued_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  control_group_act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  control_group_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  select_point_act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  select_add_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  select_unit_act_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  select_unit_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  select_worker_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  build_queue_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  unload_id_arg_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  
  fn_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  screen_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  minimap_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  screen2_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  queued_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  control_group_act_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  control_group_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_point_act_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_add_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_unit_act_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_unit_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  select_worker_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  build_queue_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  unload_id_arg_ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  
  args_sample_array = tf.TensorArray(dtype=tf.int32, size=14)
  
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  
  initial_feature_screen_shape = initial_feature_screen.shape
  initial_feature_minimap_shape = initial_feature_minimap.shape
  initial_player_shape = initial_player.shape
  initial_feature_units_shape = initial_feature_units.shape
  initial_game_loop_shape = initial_game_loop.shape
  initial_available_actions_shape = initial_available_actions.shape
  initial_build_queue_shape = initial_build_queue.shape
  initial_single_select_shape = initial_single_select.shape  
  initial_multi_select_shape = initial_multi_select.shape
  initial_score_cumulative_shape = initial_score_cumulative.shape
  initial_memory_state_shape = initial_memory_state.shape
  initial_carry_state_shape = initial_carry_state.shape
  initial_done_shape = initial_done.shape
  
  feature_screen = initial_feature_screen
  feature_minimap = initial_feature_minimap
  player = initial_player
  feature_units = initial_feature_units
  game_loop = initial_game_loop
  available_actions = initial_available_actions
  build_queue = initial_build_queue
  single_select = initial_single_select
  multi_select = initial_multi_select
  score_cumulative = initial_score_cumulative
  memory_state = initial_memory_state
  carry_state = initial_carry_state
  done = initial_done
  
  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    feature_screen = tf.expand_dims(feature_screen, 0)
    feature_minimap = tf.expand_dims(feature_minimap, 0)
    player = tf.expand_dims(player, 0)
    game_loop = tf.expand_dims(game_loop, 0)
    feature_units = tf.expand_dims(feature_units, 0)
    available_actions = tf.expand_dims(available_actions, 0)
    build_queue = tf.expand_dims(build_queue, 0)
    single_select = tf.expand_dims(single_select, 0)
    multi_select = tf.expand_dims(multi_select, 0)
    score_cumulative = tf.expand_dims(score_cumulative, 0)

    # Run the model and to get action probabilities and critic value
    model_input = {'feature_screen': feature_screen, 'feature_minimap': feature_minimap,
                   'player': player, 'feature_units': feature_units, 
                   'memory_state': memory_state, 'carry_state': carry_state, 
                   'game_loop': game_loop, 'available_actions': available_actions, 
                   'build_queue': build_queue,  'single_select': single_select, 
                   'multi_select': multi_select, 'score_cumulative': score_cumulative}

    prediction = model(model_input, training=True)
    fn_pi = prediction['fn_out']
    args_pi = prediction['args_out']
    value = prediction['value']
    memory_state = prediction['final_memory_state']
    carry_state = prediction['final_carry_state']

    fn_pi = mask_unavailable_actions(available_actions, fn_pi)
    fn_sample = sample(fn_pi)[0]
    
    args_sample = dict()
    for arg_type, arg_pi in args_pi.items():
      arg_sample = sample(arg_pi)[0]
      args_sample_array = args_sample_array.write(arg_type.id, arg_sample)
      args_sample[arg_type] = arg_sample

    # Store critic values
    values = values.write(t, tf.squeeze(value))
    
    step_result = tf_env_step(fn_sample, args_sample_array.read(0), args_sample_array.read(1), args_sample_array.read(2), 
             args_sample_array.read(3), args_sample_array.read(4), args_sample_array.read(5), args_sample_array.read(6),
             args_sample_array.read(7), args_sample_array.read(8), args_sample_array.read(9), args_sample_array.read(10),
             args_sample_array.read(11), args_sample_array.read(12))
    feature_screen = step_result[0]
    feature_minimap = step_result[1]
    player = step_result[2]
    feature_units = step_result[3] 
    game_loop = step_result[4]
    available_actions = step_result[5] 
    build_queue = step_result[6] 
    single_select = step_result[7] 
    multi_select = step_result[8] 
    score_cumulative = step_result[9] 

    reward = step_result[10]
    done = step_result[11]

    fn_id = step_result[12]
    screen_arg_id = step_result[13]
    minimap_arg_id = step_result[14]
    screen2_arg_id = step_result[15]
    queued_arg_id = step_result[16]
    control_group_act_arg_id = step_result[17]
    control_group_id_arg_id = step_result[18]
    select_point_act_arg_id = step_result[19]
    select_add_arg_id = step_result[20]
    select_unit_act_arg_id = step_result[21]
    select_unit_id_arg_id = step_result[22]
    select_worker_arg_id = step_result[23]
    build_queue_id_arg_id = step_result[24]
    unload_id_arg_id = step_result[25]
    
    fn_ids = fn_ids.write(t, fn_id)
    screen_arg_ids = screen_arg_ids.write(t, screen_arg_id)
    minimap_arg_ids = minimap_arg_ids.write(t, minimap_arg_id)
    screen2_arg_ids = screen2_arg_ids.write(t, screen2_arg_id)
    queued_arg_ids = queued_arg_ids.write(t, queued_arg_id)
    control_group_act_ids = control_group_act_ids.write(t, control_group_act_arg_id)
    control_group_id_arg_ids = control_group_id_arg_ids.write(t, control_group_id_arg_id)
    select_point_act_ids = select_point_act_ids.write(t, select_point_act_arg_id)
    select_add_arg_ids = select_add_arg_ids.write(t, select_add_arg_id)
    select_unit_act_arg_ids = select_unit_act_arg_ids.write(t, select_unit_act_arg_id)
    select_unit_id_arg_ids = select_unit_id_arg_ids.write(t, select_unit_id_arg_id)
    select_worker_arg_ids = select_worker_arg_ids.write(t, select_worker_arg_id)
    build_queue_id_arg_ids = build_queue_id_arg_ids.write(t, build_queue_id_arg_id)
    unload_id_arg_ids = unload_id_arg_ids.write(t, unload_id_arg_id)
    
    fn_probs = fn_probs.write(t, fn_pi[0, fn_id])
    for arg_type, arg_pi in args_pi.items():
      if arg_type.name == 'screen':
        screen_arg_probs = screen_arg_probs.write(t, args_pi[arg_type][0, screen_arg_id])
      elif arg_type.name == 'minimap':
        minimap_arg_probs = minimap_arg_probs.write(t, args_pi[arg_type][0, minimap_arg_id])
      elif arg_type.name == 'screen2':
        screen2_arg_probs = screen2_arg_probs.write(t, args_pi[arg_type][0, screen2_arg_id])
      elif arg_type.name == 'queued':
        queued_arg_probs = queued_arg_probs.write(t, args_pi[arg_type][0, queued_arg_id])
      elif arg_type.name == 'control_group_act':
        control_group_act_probs = control_group_act_probs.write(t, args_pi[arg_type][0, control_group_act_arg_id])
      elif arg_type.name == 'control_group_id':
        control_group_id_arg_probs = control_group_id_arg_probs.write(t, args_pi[arg_type][0, control_group_id_arg_id])
      elif arg_type.name == 'select_point_act':
        select_point_act_probs = select_point_act_probs.write(t, args_pi[arg_type][0, select_point_act_arg_id])
      elif arg_type.name == 'select_add':
        select_add_arg_probs = select_add_arg_probs.write(t, args_pi[arg_type][0, select_add_arg_id])
      elif arg_type.name == 'select_unit_act':
        select_unit_act_arg_probs = select_unit_act_arg_probs.write(t, args_pi[arg_type][0, select_unit_act_arg_id])
      elif arg_type.name == 'select_unit_id':
        select_unit_id_arg_probs = select_unit_id_arg_probs.write(t, args_pi[arg_type][0, select_unit_id_arg_id])
      elif arg_type.name == 'select_worker':
        select_worker_arg_probs = select_worker_arg_probs.write(t, args_pi[arg_type][0, select_worker_arg_id])
      elif arg_type.name == 'build_queue_id':
        build_queue_id_arg_probs = build_queue_id_arg_probs.write(t, args_pi[arg_type][0, build_queue_id_arg_id])
      elif arg_type.name == 'unload_id':
        unload_id_arg_probs = unload_id_arg_probs.write(t, args_pi[arg_type][0, unload_id_arg_id])
    
    feature_screen.set_shape(initial_feature_screen_shape)
    feature_minimap.set_shape(initial_feature_minimap_shape)
    player.set_shape(initial_player_shape)
    feature_units.set_shape(initial_feature_units_shape)
    game_loop.set_shape(initial_game_loop_shape)
    available_actions.set_shape(initial_available_actions_shape)
    build_queue.set_shape(initial_build_queue_shape)
    single_select.set_shape(initial_single_select_shape)
    multi_select.set_shape(initial_multi_select_shape)
    score_cumulative.set_shape(initial_score_cumulative_shape)
    memory_state.set_shape(initial_memory_state_shape)
    carry_state.set_shape(initial_carry_state_shape)
    
    # Store reward
    rewards = rewards.write(t, reward)
    done.set_shape(initial_done_shape)
    if tf.cast(done, tf.bool):
      break

  fn_probs = fn_probs.stack()
  screen_arg_probs = screen_arg_probs.stack()
  minimap_arg_probs = minimap_arg_probs.stack()
  screen2_arg_probs = screen2_arg_probs.stack()
  queued_arg_probs = queued_arg_probs.stack()
  control_group_act_probs = control_group_act_probs.stack()
  control_group_id_arg_probs = control_group_id_arg_probs.stack()
  select_point_act_probs = select_point_act_probs.stack()
  select_add_arg_probs = select_add_arg_probs.stack()
  select_unit_act_arg_probs = select_unit_act_arg_probs.stack()
  select_unit_id_arg_probs = select_unit_id_arg_probs.stack()
  select_worker_arg_probs = select_worker_arg_probs.stack()
  build_queue_id_arg_probs = build_queue_id_arg_probs.stack()
  unload_id_arg_probs = unload_id_arg_probs.stack()

  values = values.stack()
  rewards = rewards.stack()
  
  fn_ids = fn_ids.stack()
  screen_arg_ids = screen_arg_ids.stack()
  minimap_arg_ids = minimap_arg_ids.stack()
  screen2_arg_ids = screen2_arg_ids.stack()
  queued_arg_ids = queued_arg_ids.stack()
  control_group_act_ids = control_group_act_ids.stack()
  control_group_id_arg_ids = control_group_id_arg_ids.stack()
  select_point_act_ids = select_point_act_ids.stack()
  select_add_arg_ids = select_add_arg_ids.stack()
  select_unit_act_arg_ids = select_unit_act_arg_ids.stack()
  select_unit_id_arg_ids = select_unit_id_arg_ids.stack()
  select_worker_arg_ids = select_worker_arg_ids.stack()
  build_queue_id_arg_ids = build_queue_id_arg_ids.stack()
  unload_id_arg_ids = unload_id_arg_ids.stack()

  return (fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, control_group_act_probs, 
          control_group_id_arg_probs, select_point_act_probs, select_add_arg_probs, select_unit_act_arg_probs, 
          select_unit_id_arg_probs, select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs, 
          feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, build_queue,
          single_select, multi_select, score_cumulative, memory_state, carry_state, values, rewards, done, 
          fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, control_group_id_arg_ids,
          select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, select_unit_id_arg_ids, select_worker_arg_ids, 
          build_queue_id_arg_ids, unload_id_arg_ids
         )