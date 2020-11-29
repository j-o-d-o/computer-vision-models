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
        :params y_true: The ground truth input heat map only containing the class info (mask_height, mask_width, nb_classes)
        :params y_pred: The predicted input heat map only containing the class info (mask_height, mask_width, nb_classes)
        :return: Unweighted loss value regarding class info
        """
        pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg_mask = tf.cast(tf.less(y_true, 1.0), tf.float32)

        pos_loss = (
            -pos_mask
            * tf.math.pow(1.0 - y_pred, self.focal_loss_alpha)
            * tf.math.log(tf.clip_by_value(y_pred, 1e-4, 1. - 1e-4))
        )
        neg_loss = (
            -neg_mask
            * tf.math.pow(1.0 - y_true, self.focal_loss_beta)
            * tf.math.pow(y_pred, self.focal_loss_alpha)
            * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-4, 1. - 1e-4))
        )

        n = tf.reduce_sum(pos_mask)
        pos_loss_val = tf.reduce_sum(pos_loss)
        neg_loss_val = tf.reduce_sum(neg_loss)

        loss_val = tf.cond(tf.greater(n, 0), lambda: (pos_loss_val + neg_loss_val) / n, lambda: neg_loss_val)
        return loss_val

    def size_loss(self, y_true, y_pred, pos_mask, n):
        """
        :param y_true: The ground truth input heat map only containing the size info (mask_height, mask_width, 2)
        :param y_pred: The prediction input heat map only containing the size info (mask_height, mask_width, 2)
        :param n: Number of objects
        :return: Unweighted loss value regarding size info
        """
        loss_mask = pos_mask * (y_true - y_pred)
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(n, 0), lambda: loss_val / n, lambda: loss_val)
        return loss_val

    def offset_loss(self, y_true, y_pred, pos_mask, n):
        """
        :param y_true: The ground truth input heat map only containing the offset info (mask_height, mask_width, 2)
        :param y_pred: The prediction input heat map only containing the offset info (mask_height, mask_width, 2)
        :param n: Number of objects
        :return: Unweighted loss value regarding offset info
        """
        loss_mask = pos_mask * (y_true - y_pred)
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(n, 0), lambda: loss_val / n, lambda: loss_val)
        return loss_val

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        test_y = y_true.numpy()
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_class = y_true[:, :, :, :self.class_array_pos[1]]
        test_y_true_class = y_true_class.numpy()
        y_pred_class = y_pred[:, :, :, :self.class_array_pos[1]]

        y_true_size = y_true[:, :, :, self.size_array_pos[0]:]
        y_pred_size = y_pred[:, :, :, self.size_array_pos[0]:]
        
        y_true_offset = y_true[:, :, :, self.offset_array_pos[0]:self.offset_array_pos[1]]
        y_pred_offset = y_pred[:, :, :, self.offset_array_pos[0]:self.offset_array_pos[1]]

        class_loss = self.class_focal_loss(y_true_class, y_pred_class)

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        pos_mask = tf.reduce_max(pos_mask, axis=-1, keepdims=True)
        n = tf.reduce_sum(pos_mask)

        pos_mask_size = tf.broadcast_to(pos_mask, tf.shape(y_true_size))
        size_loss = self.size_loss(y_true_size, y_pred_size, pos_mask_size, n)

        pos_mask_offset = tf.broadcast_to(pos_mask, tf.shape(y_true_offset))
        offset_loss = self.offset_loss(y_true_offset, y_pred_offset, pos_mask_offset, n)

        total_loss = class_loss + (self.offset_weight * offset_loss) + (self.size_weight * size_loss)
        return total_loss
