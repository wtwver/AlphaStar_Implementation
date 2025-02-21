import tensorflow as tf

class ScalarArgumentHead(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarArgumentHead, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.Sequential()
    self.network.add(tf.keras.layers.Dense(output_dim, name="ScalarArgumentHead_dense_1", kernel_regularizer='l2'))
    self.network.add(tf.keras.layers.Softmax())

    self.autoregressive_embedding_encoder = tf.keras.Sequential([tf.keras.layers.Dense(self.output_dim, activation='relu', 
                                                                                name="ScalarArgumentHead_dense_2", kernel_regularizer='l2')
                                                                              ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder(autoregressive_embedding)

    network_input = tf.concat([core_output, encoded_autoregressive_embedding], axis=1)
    action_logits = self.network(network_input)
    
    return action_logits