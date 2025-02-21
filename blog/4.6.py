def run_reinforcement_episode(
    initial_state: tf.Tensor,  
    initial_reward_sum: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_logits_ts = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  
  sl_action_logits_ts = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  

  action_logits = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state
    
  reward_sum_shape = initial_reward_sum.shape
  reward_sum = initial_reward_sum
  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
  
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
    sl_action_logits_t, sl_value = sl_model(state)
  
    # Sample next action from the action probability distribution
    dist = tfd.Categorical(logits=action_logits_t[0])
    action = dist.sample()

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_logits_ts = action_logits_ts.write(t, action_logits_t[0])
    sl_action_logits_ts = sl_action_logits_ts.write(t, sl_action_logits_t[0])
    
    action_logits = action_logits.write(t, action_logits_t[0, action])
  
    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)
    
    reward_sum += int(reward)
    reward_sum.set_shape(reward_sum_shape)
  
    # Store reward
    rewards = rewards.write(t, reward)
    if tf.cast(done, tf.bool):
      break

  action_logits = action_logits.stack()
  values = values.stack()
  rewards = rewards.stack()

  action_logits_ts = action_logits_ts.stack()
  sl_action_logits_ts = sl_action_logits_ts.stack()
  
  return action_logits, values, rewards, action_logits_ts, sl_action_logits_ts