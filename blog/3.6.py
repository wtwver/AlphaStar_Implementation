optimizer = tf.keras.optimizers.Adam(lr_schedule, epsilon=1e-5)

@tf.function
def train_step(
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
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
    					 tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    prediction = run_episode(initial_feature_screen, initial_feature_minimap, initial_player, initial_feature_units, 
			      initial_game_loop, initial_available_actions, initial_build_queue, initial_single_select, 
			      initial_multi_select, initial_score_cumulative, initial_memory_state, initial_carry_state, 
			      initial_done, model, max_steps_per_episode) 
    fn_probs = prediction[0] 
    screen_arg_probs = prediction[1]  
    minimap_arg_probs = prediction[2] 
    screen2_arg_probs = prediction[3] 
    queued_arg_probs = prediction[4] 
    control_group_act_probs = prediction[5] 
    control_group_id_arg_probs = prediction[6] 
    select_point_act_probs = prediction[7] 
    select_add_arg_probs = prediction[8] 
    select_unit_act_arg_probs = prediction[9] 
    select_unit_id_arg_probs = prediction[10]
    select_worker_arg_probs = prediction[11] 
    build_queue_id_arg_probs = prediction[12] 
    unload_id_arg_probs = prediction[13]
    
    feature_screen = prediction[14]
    feature_minimap = prediction[15] 
    player = prediction[16]
    feature_units = prediction[17] 
    game_loop = prediction[18]
    available_actions = prediction[19]
    
    build_queue = prediction[20]
    single_select = prediction[21]
    multi_select = prediction[22]
    score_cumulative = prediction[23]

    memory_state = prediction[24]
    carry_state = prediction[25]
    
    values = prediction[26]
    rewards = prediction[27]
    done = prediction[28]

    fn_ids = prediction[29]
    screen_arg_ids = prediction[30]
    minimap_arg_ids = prediction[31] 
    screen2_arg_ids = prediction[32] 
    queued_arg_ids = prediction[33] 
    control_group_act_ids = prediction[34] 
    control_group_id_arg_ids = prediction[35]
    select_point_act_ids = prediction[36]
    select_add_arg_ids = prediction[37]
    select_unit_act_arg_ids = prediction[38] 
    select_unit_id_arg_ids = prediction[39] 
    select_worker_arg_ids = prediction[40] 
    build_queue_id_arg_ids = prediction[41] 
    unload_id_arg_ids = prediction[42]
    
    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)
    
    # Convert training data to appropriate TF tensor shapes
    converted_value = [
        tf.expand_dims(x, 1) for x in [fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs, queued_arg_probs, 
        				control_group_act_probs, control_group_id_arg_probs, select_point_act_probs, 
        				select_add_arg_probs, select_unit_act_arg_probs, select_unit_id_arg_probs, 
        				select_worker_arg_probs, build_queue_id_arg_probs, unload_id_arg_probs, values, returns,
        				fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids,
        				control_group_id_arg_ids, select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids,
        				select_unit_id_arg_ids, select_worker_arg_ids, build_queue_id_arg_ids, unload_id_arg_ids
        				]] 
    fn_probs = converted_value[0]
    screen_arg_probs = converted_value[1] 
    minimap_arg_probs = converted_value[2] 
    screen2_arg_probs = converted_value[3] 
    queued_arg_probs = converted_value[4]
    control_group_act_probs = converted_value[5] 
    control_group_id_arg_probs = converted_value[6] 
    select_point_act_probs = converted_value[7]
    select_add_arg_probs = converted_value[8] 
    select_unit_act_arg_probs = converted_value[9] 
    select_unit_id_arg_probs = converted_value[10] 
    select_worker_arg_probs = converted_value[11] 
    build_queue_id_arg_probs = converted_value[12] 
    unload_id_arg_probs = converted_value[13]
    
    values = converted_value[14]
    returns = converted_value[15]
    
    fn_ids = converted_value[16]
    screen_arg_ids = converted_value[17]
    minimap_arg_ids = converted_value[18]
    screen2_arg_ids = converted_value[19]
    queued_arg_ids = converted_value[20]
    control_group_act_ids = converted_value[21]
    control_group_id_arg_ids = converted_value[22]
    select_point_act_ids = converted_value[23]
    select_add_arg_ids = converted_value[24]
    select_unit_act_arg_ids = converted_value[25]
    select_unit_id_arg_ids = converted_value[26]
    select_worker_arg_ids = converted_value[27]
    build_queue_id_arg_ids = converted_value[28]
    unload_id_arg_ids = converted_value[29]

    # Calculating loss values to update our network
    loss = compute_loss(fn_probs, screen_arg_probs, minimap_arg_probs, screen2_arg_probs,
			 queued_arg_probs, control_group_act_probs, control_group_id_arg_probs, select_point_act_probs, 
			 select_add_arg_probs, select_unit_act_arg_probs, select_unit_id_arg_probs, select_worker_arg_probs, 
			 build_queue_id_arg_probs, unload_id_arg_probs, values, returns,
			 fn_ids, screen_arg_ids, minimap_arg_ids, screen2_arg_ids, queued_arg_ids, control_group_act_ids, 
			 control_group_id_arg_ids, select_point_act_ids, select_add_arg_ids, select_unit_act_arg_ids, 
			 select_unit_id_arg_ids, select_worker_arg_ids, build_queue_id_arg_ids, unload_id_arg_ids
			)
  
  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)
  grad_norm = tf.linalg.global_norm(grads)
  grads, _ = tf.clip_by_global_norm(grads, arguments.gradient_clipping)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return (episode_reward, feature_screen, feature_minimap, player, feature_units, game_loop, available_actions, 
          build_queue, single_select, multi_select, score_cumulative, memory_state, carry_state, done, grad_norm)