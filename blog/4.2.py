cce_loss = tf.keras.losses.CategoricalCrossentropy()

def compute_supervised_loss(
    action_probs: tf.Tensor,  
    actions: tf.Tensor) -> tf.Tensor:
  """Computes the supervised loss."""

  actions_onehot = tf.one_hot(actions, num_actions)
  action_loss = cce_loss(actions_onehot, action_probs)

  return action_loss