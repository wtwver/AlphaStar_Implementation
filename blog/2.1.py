import tensorflow as tf

class SpatialEncoder(tf.keras.layers.Layer):
  def __init__(self, height, width, channel):
    super(SpatialEncoder, self).__init__()

    self.height = height
    self.width = width
    self.channel = channel

    self.network = tf.keras.Sequential([
       tf.keras.layers.Conv2D(self.channel, 1, padding='same', activation='relu', name="SpatialEncoder_cond2d_1", 
                                   kernel_regularizer='l2'),
       tf.keras.layers.Conv2D(self.channel, 5, padding='same', activation='relu', name="SpatialEncoder_cond2d_2", 
                                   kernel_regularizer='l2'),
       tf.keras.layers.Conv2D(self.channel*2, 3, padding='same', activation='relu', name="SpatialEncoder_cond2d_3", 
                                   kernel_regularizer='l2')
    ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
        'channel': self.channel
    })
    return config

  def call(self, spatial_feature):
    spatial_feature_encoded = self.network(spatial_feature)

    return spatial_feature_encoded