mse_loss = tf.keras.losses.MeanSquaredError()

def compute_loss(
    fn_probs: tf.Tensor, screen_arg_probs: tf.Tensor, minimap_arg_probs: tf.Tensor, screen2_arg_probs: tf.Tensor, 
    queued_arg_probs: tf.Tensor, control_group_act_probs: tf.Tensor, control_group_id_arg_probs: tf.Tensor, 
    select_point_act_probs: tf.Tensor, select_add_arg_probs: tf.Tensor, select_unit_act_arg_probs: tf.Tensor, 
    select_unit_id_arg_probs: tf.Tensor, select_worker_arg_probs: tf.Tensor, build_queue_id_arg_probs: tf.Tensor,
    unload_id_arg_probs: tf.Tensor,
    values: tf.Tensor, returns: tf.Tensor,
    fn_ids: tf.Tensor, screen_arg_ids: tf.Tensor, minimap_arg_ids: tf.Tensor, screen2_arg_ids: tf.Tensor, 
    queued_arg_ids: tf.Tensor, control_group_act_ids: tf.Tensor, control_group_id_arg_ids: tf.Tensor, 
    select_point_act_ids: tf.Tensor, select_add_arg_ids: tf.Tensor, select_unit_act_arg_ids: tf.Tensor, 
    select_unit_id_arg_ids: tf.Tensor, select_worker_arg_ids: tf.Tensor, build_queue_id_arg_ids: tf.Tensor, 
    unload_id_arg_ids: tf.Tensor
    ) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values
  
  fn_log_probs = tf.math.log(fn_probs)
  screen_arg_log_probs = tf.math.log(screen_arg_probs) * tf.cast(tf.not_equal(screen_arg_ids, -1), 'float32')
  minimap_arg_log_probs = tf.math.log(minimap_arg_probs) * tf.cast(tf.not_equal(minimap_arg_ids, -1), 'float32')
  screen2_arg_log_probs = tf.math.log(screen2_arg_probs) * tf.cast(tf.not_equal(screen2_arg_ids, -1), 'float32')
  queued_arg_log_probs = tf.math.log(queued_arg_probs) * tf.cast(tf.not_equal(queued_arg_ids, -1), 'float32')
  control_group_act_log_probs = tf.math.log(control_group_act_probs) * tf.cast(tf.not_equal(control_group_act_ids, -1), 'float32')
  control_group_id_arg_log_probs = tf.math.log(control_group_id_arg_probs) * tf.cast(tf.not_equal(control_group_id_arg_ids, -1), 'float32')
  select_point_act_log_probs = tf.math.log(select_point_act_probs) * tf.cast(tf.not_equal(select_point_act_ids, -1), 'float32')
  select_add_arg_log_probs = tf.math.log(select_add_arg_probs) * tf.cast(tf.not_equal(select_add_arg_ids, -1), 'float32')
  select_unit_act_arg_log_probs = tf.math.log(select_unit_act_arg_probs) * tf.cast(tf.not_equal(select_unit_act_arg_ids, -1), 'float32')
  select_unit_id_arg_log_probs = tf.math.log(select_unit_id_arg_probs) * tf.cast(tf.not_equal(select_unit_id_arg_ids, -1), 'float32')
  select_worker_arg_log_probs = tf.math.log(select_worker_arg_probs) * tf.cast(tf.not_equal(select_worker_arg_ids, -1), 'float32')
  build_queue_id_arg_log_probs = tf.math.log(build_queue_id_arg_probs) * tf.cast(tf.not_equal(build_queue_id_arg_ids, -1), 'float32')
  unload_id_arg_log_probs = tf.math.log(unload_id_arg_probs) * tf.cast(tf.not_equal(unload_id_arg_ids, -1), 'float32')
  
  log_probs = fn_log_probs + screen_arg_log_probs + minimap_arg_log_probs + screen2_arg_log_probs + queued_arg_log_probs + control_group_act_log_probs + \
  		control_group_id_arg_log_probs + select_point_act_log_probs + select_add_arg_log_probs + select_unit_act_arg_log_probs + \
  		  select_unit_id_arg_log_probs + select_worker_arg_log_probs + build_queue_id_arg_log_probs + unload_id_arg_log_probs
  
  actor_loss = -tf.math.reduce_mean(log_probs * tf.stop_gradient(advantage))
  critic_loss = mse_loss(values, returns)

  entropy = -tf.reduce_sum(fn_log_probs * fn_probs, axis=-1)
  entropy += -tf.reduce_sum(screen_arg_log_probs * screen_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(minimap_arg_log_probs * minimap_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(screen2_arg_log_probs * screen2_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(queued_arg_log_probs * queued_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(control_group_act_log_probs * control_group_act_probs, axis=-1)
  entropy += -tf.reduce_sum(control_group_id_arg_log_probs * control_group_id_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(select_point_act_log_probs * select_point_act_probs, axis=-1)
  entropy += -tf.reduce_sum(select_add_arg_log_probs * select_add_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(select_unit_act_arg_log_probs * select_unit_act_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(select_unit_id_arg_log_probs * select_unit_id_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(select_worker_arg_log_probs * select_worker_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(build_queue_id_arg_log_probs * build_queue_id_arg_probs, axis=-1)
  entropy += -tf.reduce_sum(unload_id_arg_log_probs * unload_id_arg_probs, axis=-1)\
  
  entropy_loss = tf.reduce_mean(entropy)

  return actor_loss + 0.5 * critic_loss - 1e-3 * entropy_loss