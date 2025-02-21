supervised_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

@tf.function
def train_supervised_step(
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    # Run the model for one episode to collect training data
    action_probs, actions = run_supervised_episode(model) 

    # Convert training data to appropriate TF tensor shapes
    action_probs, actions = [tf.expand_dims(x, 1) for x in [action_probs, actions]] 

    # Calculating loss values to update our network
    loss = compute_supervised_loss(action_probs, actions)
    
    regularization_loss = tf.reduce_sum(model.losses)
    loss = loss + regularization_loss

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  supervised_optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
  return loss