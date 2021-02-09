import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams


class DmdsLoss(Loss):
    def __init__(self, params: DmdsParams):
        super().__init__()
        self.params = params

    def obj_dims_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.obj_dims_pos[0]:self.obj_dims_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.obj_dims_pos[0]:self.obj_dims_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def calc_loss(self, y_true, y_true_feat, y_pred_feat, loss_type: str = "mse"):
        y_true_class = y_true[:, :, :, :self.class_pos[1]]

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        pos_mask = tf.reduce_max(pos_mask, axis=-1, keepdims=True)
        pos_mask_feat = tf.broadcast_to(pos_mask, tf.shape(y_true_feat))
        nb_objects = tf.reduce_sum(pos_mask)

        if loss_type == "mse":
            loss_mask = pos_mask_feat * tf.math.squared_difference(y_true_feat, y_pred_feat)
        elif loss_type == "mae":
            loss_mask = pos_mask_feat * (y_true_feat - y_pred_feat)
        elif loss_type == "mape":
            loss_mask = pos_mask_feat * ((y_true_feat - y_pred_feat) / tf.maximum(tf.math.abs(y_true_feat), 1.0))
        else:
            assert(False)
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(nb_objects, 0), lambda: loss_val / nb_objects, lambda: loss_val)
        return loss_val

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        total_loss = 0

        return total_loss
