def env_step(fn_sample: np.ndarray, screen_arg_sample: np.ndarray, minimap_arg_sample: np.ndarray, screen2_arg_sample: np.ndarray, 
       queued_arg_sample: np.ndarray, control_group_act_arg_sample: np.ndarray, control_group_id_arg_sample: np.ndarray, 
       select_point_act_arg_sample: np.ndarray, select_add_arg_sample: np.ndarray, select_unit_act_arg_sample: np.ndarray,
       select_unit_id_arg_sample: np.ndarray, select_worker_arg_sample: np.ndarray, build_queue_id_arg_sample: np.ndarray,
       unload_id_arg_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  args_sample = dict()
  for arg_type in actions.TYPES:
    if arg_type.name == 'screen':
      args_sample[arg_type] = screen_arg_sample
    elif arg_type.name == 'minimap':
      args_sample[arg_type] = minimap_arg_sample
    elif arg_type.name == 'screen2':
      args_sample[arg_type] = screen2_arg_sample
    elif arg_type.name == 'queued':
      args_sample[arg_type] = queued_arg_sample
    elif arg_type.name == 'control_group_act':
      args_sample[arg_type] = control_group_act_arg_sample
    elif arg_type.name == 'control_group_id':
      args_sample[arg_type] = control_group_id_arg_sample
    elif arg_type.name == 'select_point_act':
      args_sample[arg_type] =select_point_act_arg_sample
    elif arg_type.name == 'select_add':
      args_sample[arg_type] = select_add_arg_sample
    elif arg_type.name == 'select_unit_act':
      args_sample[arg_type] = select_unit_act_arg_sample
    elif arg_type.name == 'select_unit_id':
      args_sample[arg_type] = select_unit_id_arg_sample
    elif arg_type.name == 'select_worker':
      args_sample[arg_type] = select_worker_arg_sample
    elif arg_type.name == 'build_queue_id':
      args_sample[arg_type] = build_queue_id_arg_sample
    elif arg_type.name == 'unload_id':
      args_sample[arg_type] = unload_id_arg_sample

  fn_id, args_id = mask_unused_argument_samples(fn_sample, args_sample)

  arg_id_list = []
  for arg_type in actions.TYPES:
    arg_id_list.append(args_id[arg_type])

  actions_list = actions_to_pysc2(fn_id, args_id, (32, 32))
  actions_list = [actions_list]

  next_state = env.step(actions_list)
  next_state = next_state[0]
  done = next_state[0]
  if done == StepType.LAST:
    done = True
  else:
    done = False
  
  reward = float(next_state[1])
  
  feature_screen = next_state[3]['feature_screen']
  feature_screen = utils.preprocess_screen(feature_screen)
  feature_screen = np.transpose(feature_screen, (1, 2, 0))
  
  feature_minimap = next_state[3]['feature_minimap']
  feature_minimap = utils.preprocess_minimap(feature_minimap)
  feature_minimap = np.transpose(feature_minimap, (1, 2, 0))
    
  player = next_state[3]['player']
  player = utils.preprocess_player(player)
    
  available_actions = next_state[3]['available_actions']
  available_actions = utils.preprocess_available_actions(available_actions)

  feature_units = next_state[3]['feature_units']
  feature_units = utils.preprocess_feature_units(feature_units, 32)
    
  game_loop = next_state[3]['game_loop']
  
  build_queue = next_state[3]['build_queue']
  build_queue = utils.preprocess_build_queue(build_queue)

  single_select = next_state[3]['single_select']
  single_select = utils.preprocess_single_select(single_select)

  multi_select = next_state[3]['multi_select']
  multi_select = utils.preprocess_multi_select(multi_select)

  score_cumulative = next_state[3]['score_cumulative']
  score_cumulative = utils.preprocess_score_cumulative(score_cumulative)

  return (feature_screen.astype(np.float32), feature_minimap.astype(np.float32), player.astype(np.float32), 
           feature_units.astype(np.float32), game_loop.astype(np.int32), available_actions.astype(np.int32), 
           build_queue.astype(np.float32), single_select.astype(np.float32), multi_select.astype(np.float32), 
           score_cumulative.astype(np.float32), np.array(reward, np.float32), np.array(done, np.float32), np.array(fn_id, np.int32),
           np.array(arg_id_list[0], np.int32), np.array(arg_id_list[1], np.int32), np.array(arg_id_list[2], np.int32), 
           np.array(arg_id_list[3], np.int32), np.array(arg_id_list[4], np.int32), np.array(arg_id_list[5], np.int32), 
           np.array(arg_id_list[6], np.int32), np.array(arg_id_list[7], np.int32), np.array(arg_id_list[8], np.int32), 
           np.array(arg_id_list[9], np.int32), np.array(arg_id_list[10], np.int32), np.array(arg_id_list[11], np.int32),
           np.array(arg_id_list[12], np.int32)
           )


def tf_env_step(fn_id: tf.Tensor, screen_arg_id: tf.Tensor, minimap_arg_id: tf.Tensor, screen2_arg_id: tf.Tensor, 
          queued_arg_id: tf.Tensor, control_group_act_arg_id: tf.Tensor, control_group_id_arg_id: tf.Tensor, 
          select_point_act_arg_id: tf.Tensor, select_add_arg_id: tf.Tensor, select_unit_act_arg_id: tf.Tensor, 
          select_unit_id_arg_id: tf.Tensor, select_worker_arg_id: tf.Tensor, build_queue_id_arg_id: tf.Tensor,
          unload_id_arg_id: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [fn_id, screen_arg_id, minimap_arg_id, screen2_arg_id, queued_arg_id, 
            control_group_act_arg_id, control_group_id_arg_id, select_point_act_arg_id, select_add_arg_id, 
            select_unit_act_arg_id, select_unit_id_arg_id, select_worker_arg_id, build_queue_id_arg_id, 
            unload_id_arg_id], 
                           [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.float32,
                            tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
                            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])