optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

@tf.function
def train_rl_step(
    initial_state: tf.Tensor, 
    initial_reward_sum: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_logits, values, rewards, action_logits_ts, sl_action_logits_ts = run_reinforcement_episode(
        initial_state, initial_reward_sum, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_logits, values, returns = [
        tf.expand_dims(x, 1) for x in [action_logits, values, returns]] 

    # Calculating loss values to update our network
    rl_loss = compute_rl_loss(action_logits, values, returns)
    kl_loss = compute_kl_loss(action_logits_ts, sl_action_logits_ts)
    
    loss = rl_loss + kl_loss

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward