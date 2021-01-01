import tensorflow as tf

class BatchNormalization_Quant(tf.keras.layers.BatchNormalization):
  def call(self, inputs, training = None):
    # Loss beta
    beta = self.beta
    num_beta = tf.dtypes.cast(tf.size(beta), tf.float32)
    int_values_beta = tf.clip_by_value(tf.math.round(beta), -128, 127)
    abs_loss_beta = tf.reduce_sum(tf.math.square(beta - int_values_beta))
    # Loss gamma
    gamma = self.gamma
    num_gamma = tf.dtypes.cast(tf.size(gamma), tf.float32)
    int_values_gamma = tf.clip_by_value(tf.math.round(gamma), -128, 127)
    abs_loss_gamma = tf.reduce_sum(tf.math.square(gamma - int_values_gamma))
    # Total loss
    normalized_loss = (abs_loss_beta + abs_loss_gamma) / ((num_beta + num_gamma) * 0.25 * 1.5)
    self.add_loss(normalized_loss)

    return super().call(inputs, training)
