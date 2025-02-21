kl = tf.keras.losses.KLDivergence()

def compute_kl_loss(
    action_logits_ts: tf.Tensor,  
    sl_action_logits_ts: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  dist = tfd.Categorical(logits=action_logits_ts)
  sl_dist = tfd.Categorical(logits=sl_action_logits_ts) 

  kl_loss = tfd.kl_divergence(dist, sl_dist)
  kl_loss = tf.reduce_mean(kl_loss)

  return kl_loss