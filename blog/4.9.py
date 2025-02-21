def supervised_replay(self, replay_feature_screen_list, replay_feature_minimap_list,
                      replay_feature_player_list, replay_feature_units_list, 
                      replay_available_actions_list, replay_fn_id_list, replay_args_ids_list,
                      memory_state_list, carry_state_list,
                      replay_game_loop_list, last_action_type_list,
                      replay_build_queue_list, replay_single_select_list, replay_multi_select_list,
                      replay_score_cumulative_list):
  replay_feature_screen_array = tf.concat(replay_feature_screen_list, 0)
  replay_feature_minimap_array = tf.concat(replay_feature_minimap_list, 0)
  replay_feature_player_array = tf.concat(replay_feature_player_list, 0)
  replay_feature_units_array = tf.concat(replay_feature_units_list, 0)
  replay_memory_state_array = tf.concat(memory_state_list, 0)
  replay_carry_state_array = tf.concat(carry_state_list, 0)
  replay_game_loop_array = tf.concat(replay_game_loop_list, 0)
  last_action_type_array = tf.concat(last_action_type_list, 0)
  replay_available_actions_array = tf.concat(replay_available_actions_list, 0)
  replay_fn_id_array = tf.concat(replay_fn_id_list, 0)
  replay_arg_ids_array = tf.concat(replay_args_ids_list, 0)

  replay_build_queue_array = tf.concat(replay_build_queue_list, 0)
  replay_single_select_array = tf.concat(replay_single_select_list, 0)
  replay_multi_select_array = tf.concat(replay_multi_select_list, 0)
  replay_score_cumulative_array = tf.concat(replay_score_cumulative_list, 0)

  with tf.GradientTape() as tape:
    input_ = {'feature_screen': replay_feature_screen_array, 'feature_minimap': replay_feature_minimap_array,
              'feature_player': replay_feature_player_array, 'feature_units': replay_feature_units_array, 
              'memory_state': replay_memory_state_array, 'carry_state': replay_carry_state_array, 
              'game_loop': replay_game_loop_array, 'available_actions': replay_available_actions_array, 
              'last_action_type': last_action_type_array, 'build_queue': replay_build_queue_array, 
              'single_select': replay_single_select_array, 'multi_select': replay_multi_select_array, 
              'score_cumulative': replay_score_cumulative_array}
    prediction = self.ActorCritic(input_, training=True)
    fn_pi = prediction['fn_out']
    arg_pis = prediction['args_out']
    next_memory_state = prediction['final_memory_state']
    next_carry_state = prediction['final_carry_state']

    batch_size = fn_pi.shape[0]

    replay_fn_id_array_onehot = tf.one_hot(replay_fn_id_array, 573)
    replay_fn_id_array_onehot = tf.reshape(replay_fn_id_array_onehot, (batch_size, 573)

    replay_fn_id_array_onehot *= replay_available_actions_array

    fn_id_loss = cce(replay_fn_id_array_onehot, fn_pi)
    arg_ids_loss = 0 
    for index, arg_type in enumerate(actions.TYPES):
      replay_arg_id = replay_arg_ids_array[:,index]
      arg_pi = arg_pis[arg_type]
      replay_arg_id_array_onehot = tf.one_hot(replay_arg_id, arg_pi.shape[1])

      arg_id_loss = cce(replay_arg_id_array_onehot, arg_pi)
      arg_ids_loss += arg_id_loss

      regularization_loss = tf.reduce_sum(self.ActorCritic.losses)

      total_loss = fn_id_loss + arg_ids_loss + 1e-5 * regularization_loss

    grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
    self.optimizer_sl.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))