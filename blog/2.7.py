class Baseline(tf.keras.layers.Layer):
  def __init__(self, output_dim):
    super(Baseline, self).__init__()

    self.output_dim = output_dim
    self.network = tf.keras.Sequential([tf.keras.layers.Dense(1, name="Baseline_dense", kernel_regularizer='l2')])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'output_dim': self.output_dim
    })
    return config

  def call(self, core_output):
    batch_size = tf.shape(core_output)[0]
    network_input = core_output
    value = self.network(network_input)

    return value