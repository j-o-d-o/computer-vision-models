import tensorflow as tf
from tensorflow.keras.losses import Loss


class Centernet2DLoss(Loss):
    def __init__(self, nb_classes, size_weight, offset_weight, focal_loss_alpha, focal_loss_beta):
        super().__init__()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_beta = focal_loss_beta
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.class_array_pos = (0, nb_classes)
        self.offset_array_pos = (nb_classes, nb_classes + 2)
        self.size_array_pos = (nb_classes + 2, None)

    def class_focal_loss(self, y_true, y_pred):
        """
        :params y_true: ground truth
        :params y_pred: prediction
        :return: Unweighted loss value regarding class info
        """
        y_true_class = y_true[:, :, :, :self.class_array_pos[1]]
        y_pred_class = y_pred[:, :, :, :self.class_array_pos[1]]

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        neg_mask = tf.cast(tf.less(y_true_class, 1.0), tf.float32)

        pos_loss = (
            -pos_mask
            * tf.math.pow(1.0 - y_pred_class, self.focal_loss_alpha)
            * tf.math.log(tf.clip_by_value(y_pred_class, 1e-4, 1. - 1e-4))
        )
        neg_loss = (
            -neg_mask
            * tf.math.pow(1.0 - y_true_class, self.focal_loss_beta)
            * tf.math.pow(y_pred_class, self.focal_loss_alpha)
            * tf.math.log(tf.clip_by_value(1.0 - y_pred_class, 1e-4, 1. - 1e-4))
        )

        n = tf.reduce_sum(pos_mask)
        pos_loss_val = tf.reduce_sum(pos_loss)
        neg_loss_val = tf.reduce_sum(neg_loss)

        loss_val = tf.cond(tf.greater(n, 0), lambda: (pos_loss_val + neg_loss_val) / n, lambda: neg_loss_val)
        return loss_val

    def size_loss(self, y_true, y_pred):
        """
        :params y_true: ground truth
        :params y_pred: prediction
        :return: Unweighted loss value regarding size info
        """
        y_true_class = y_true[:, :, :, :self.class_array_pos[1]]

        y_true_size = y_true[:, :, :, self.size_array_pos[0]:]
        y_pred_size = y_pred[:, :, :, self.size_array_pos[0]:]

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        pos_mask = tf.reduce_max(pos_mask, axis=-1, keepdims=True)
        pos_mask_size = tf.broadcast_to(pos_mask, tf.shape(y_true_size))
        nb_objects = tf.reduce_sum(pos_mask)

        loss_mask = pos_mask_size * (y_true_size - y_pred_size)
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(nb_objects, 0), lambda: loss_val / nb_objects, lambda: loss_val)
        return loss_val

    def offset_loss(self, y_true, y_pred):
        """
        :params y_true: ground truth
        :params y_pred: prediction
        """
        y_true_class = y_true[:, :, :, :self.class_array_pos[1]]

        y_true_offset = y_true[:, :, :, self.offset_array_pos[0]:self.offset_array_pos[1]]
        y_pred_offset = y_pred[:, :, :, self.offset_array_pos[0]:self.offset_array_pos[1]]

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        pos_mask = tf.reduce_max(pos_mask, axis=-1, keepdims=True)
        pos_mask_offset = tf.broadcast_to(pos_mask, tf.shape(y_true_offset))
        nb_objects = tf.reduce_sum(pos_mask)

        loss_mask = pos_mask_offset * (y_true_offset - y_pred_offset)
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(nb_objects, 0), lambda: loss_val / nb_objects, lambda: loss_val)
        return loss_val

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        class_loss = self.class_focal_loss(y_true, y_pred)
        size_loss = self.size_loss(y_true, y_pred)
        offset_loss = self.offset_loss(y_true, y_pred)

        total_loss = class_loss + (self.offset_weight * offset_loss) + (self.size_weight * size_loss)
        return total_loss
