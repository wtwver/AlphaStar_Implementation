import tensorflow as tf

class ScalarEncoder(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(ScalarEncoder, self).__init__()
    self.output_dim = output_dim

    self.network = tf.keras.Sequential([
       tf.keras.layers.Dense(self.output_dim, activation='relu', name="ScalarEncoder_dense", kernel_regularizer='l2')
    ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, scalar_feature):
    batch_size = tf.shape(scalar_feature)[0]
    scalar_feature_encoded = self.network(scalar_feature)
    return scalar_feature_encoded