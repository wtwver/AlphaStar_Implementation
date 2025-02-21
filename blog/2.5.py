import tensorflow as tf

class SpatialArgumentHead(tf.keras.layers.Layer):
  def __init__(self, height, width):
    super(SpatialArgumentHead, self).__init__()

    self.height = height
    self.width = width
    self.network = tf.keras.Sequential([tf.keras.layers.Conv2D(1, 1, padding='same', name="SpatialArgumentHead_conv2d_1", 
                                                                            kernel_regularizer='l2'),
                                                tf.keras.layers.Flatten(),
                                                tf.keras.layers.Softmax()])

    self.autoregressive_embedding_encoder_1 = tf.keras.Sequential([tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                          name="SpatialArgumentHead_dense_1", kernel_regularizer='l2')])
    self.autoregressive_embedding_encoder_2 = tf.keras.Sequential([tf.keras.layers.Dense(self.height * self.width, activation='relu', 
                                                                          name="SpatialArgumentHead_dense_2", kernel_regularizer='l2')])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
    })
    return config

  def call(self, feature_encoded, core_output, autoregressive_embedding):
    batch_size = tf.shape(core_output)[0]

    encoded_core_output = self.autoregressive_embedding_encoder_1(core_output)
    encoded_core_output = tf.reshape(encoded_core_output, (batch_size, self.height, self.width, 1))

    encoded_autoregressive_embedding = self.autoregressive_embedding_encoder_2(autoregressive_embedding)
    encoded_autoregressive_embedding = tf.reshape(encoded_autoregressive_embedding, (batch_size, self.height, self.width, 1))

    network_input = tf.concat([feature_encoded, encoded_core_output, encoded_autoregressive_embedding], axis=3)
    action_logits = self.network(network_input)

    return action_logits