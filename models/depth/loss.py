import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.depth.params import Params
import matplotlib.pyplot as plt


class DepthLoss(Loss):
    def call(self, y_true, y_pred):
        # All 0 values should not contribute, but network should not learn this mask
        pos_mask = tf.cast(tf.greater(y_true, 1.0), tf.float32)
        n = tf.reduce_sum(pos_mask)
        y_true = y_true + 1e-10
        y_pred = y_pred + 1e-10
        disp_true = tf.math.reciprocal_no_nan(y_true)
        disp_pred = tf.math.reciprocal_no_nan(y_pred)
        loss_val = 0
        loss_val = pos_mask * (tf.squeeze(disp_pred) - disp_true)
        # loss_val = pos_mask * tf.math.squared_difference(tf.squeeze(y_pred), y_true)
        loss_val = tf.math.abs(loss_val)
        loss_val = tf.reduce_sum(loss_val)
        loss_val = tf.cond(tf.greater(n, 0), lambda: (loss_val) / n, lambda: loss_val)
        return loss_val
