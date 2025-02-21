import tensorflow as tf

class Core(tf.keras.layers.Layer):
  def __init__(self, unit_number, network_scale):
    super(Core, self).__init__()

    self.unit_number = unit_number
    self.network_scale = network_scale

    self.lstm_1 = LSTM(256*self.network_scale*self.network_scale, name="core_lstm_1", return_sequences=True, 
                          return_state=True, kernel_regularizer='l2')
    self.lstm_2 = LSTM(256*self.network_scale*self.network_scale, name="core_lstm_2", return_sequences=True, 
                          return_state=True, kernel_regularizer='l2')

    self.network = tf.keras.Sequential([Reshape((1068, 256*self.network_scale*self.network_scale)),
                                                Flatten(),
                                                tf.keras.layers.Dense(256*self.network_scale*self.network_scale, activation='relu', 
                                                                           name="core_dense", 
                                                                           kernel_regularizer='l2')
                                           ])

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'unit_number': self.unit_number,
        'network_scale': self.network_scale
    })
    return config

  def call(self, feature_encoded, memory_state, carry_state, training=False):
    batch_size = tf.shape(feature_encoded)[0]

    feature_encoded_flattened = Flatten()(feature_encoded)
    feature_encoded_flattened = Reshape((1068, 256*self.network_scale*self.network_scale))(feature_encoded_flattened)

    initial_state_1 = (memory_state, carry_state)
    core_output_1, final_memory_state_1, final_carry_state_1 = self.lstm_1(feature_encoded_flattened, 
                                                                                         initial_state=initial_state_1, 
                                                                                         training=training)
    initial_state_2 = (final_memory_state_1, final_memory_state_1)
    core_output_2, final_memory_state_2, final_carry_state_2 = self.lstm_2(core_output_1, initial_state=initial_state_2, 
                                                                                         training=training)

    core_output = self.network(core_output_2)

    return core_output, final_memory_state_2, final_carry_state_2
