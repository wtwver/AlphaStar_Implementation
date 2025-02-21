import tensorflow as tf

class ActionTypeHead(tf.keras.layers.Layer):
  def __init__(self, output_dim, network_scale):
    super(ActionTypeHead, self).__init__()

    self.output_dim = output_dim
    self.network_scale = network_scale
    self.network = tf.keras.Sequential([tf.keras.layers.Dense(self.output_dim, name="ActionTypeHead_dense_1", kernel_regularizer='l2'),
                                                tf.keras.layers.Softmax()])
    self.autoregressive_embedding_network = tf.keras.Sequential([tf.keras.layers.Dense(256*self.network_scale*self.network_scale, 
                                                                                                 activation='relu', name="ActionTypeHead_dense_2", 
                                                                                                 kernel_regularizer='l2')])
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim,
        'network_scale': self.network_scale
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]
    action_type_logits = self.network(core_output)

    action_type_dist = tfd.Categorical(probs=action_type_logits)
    action_type = action_type_dist.sample()
    action_type_onehot = tf.one_hot(action_type, self.output_dim)

    autoregressive_embedding = self.autoregressive_embedding_network(action_type_onehot)
    autoregressive_embedding += core_output

    return action_type_logits, autoregressive_embedding