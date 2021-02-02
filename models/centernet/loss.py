import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.centernet.params import CenternetParams


class CenternetLoss(Loss):
    def __init__(self, params: CenternetParams):
        super().__init__()
        self.params = params

        # pre calc position of data within the y_true and y_pred
        self.class_pos = [0, params.NB_CLASSES]
        if params.REGRESSION_FIELDS["r_offset"].active:
            self.r_offset_pos = [params.start_idx("r_offset"), params.end_idx("r_offset")]
        if params.REGRESSION_FIELDS["fullbox"].active:
            self.fullbox_pos = [params.start_idx("fullbox"), params.end_idx("fullbox")]
        if params.REGRESSION_FIELDS["l_shape"].active:
            # TODO: Split up the values in left, right edge and bottom center. Because bottom center
            #       should have reduced loss when it is on/or close to the line between left and right edge point
            #       because then the bottom center could be anywhere on the line, it doesnt really matter much
            self.l_shape_pos = [params.start_idx("l_shape"), params.end_idx("l_shape")]
        if params.REGRESSION_FIELDS["3d_info"].active:
            # these are split up as they have different loss functions
            start_idx = params.start_idx("3d_info")
            self.radial_dist_pos = start_idx
            self.orientation_pos = start_idx + 1
            self.obj_dims_pos = [start_idx + 2, start_idx + 5]

    def class_focal_loss(self, y_true, y_pred):
        y_true_class = y_true[:, :, :, :self.class_pos[1]]
        y_pred_class = y_pred[:, :, :, :self.class_pos[1]]

        pos_mask = tf.cast(tf.equal(y_true_class, 1.0), tf.float32)
        neg_mask = tf.cast(tf.less(y_true_class, 1.0), tf.float32)

        pos_loss = (
            -pos_mask
            * tf.math.pow(1.0 - y_pred_class, self.params.FOCAL_LOSS_ALPHA)
            * tf.math.log(tf.clip_by_value(y_pred_class, 1e-4, 1. - 1e-4))
        )
        neg_loss = (
            -neg_mask
            * tf.math.pow(1.0 - y_true_class, self.params.FOCAL_LOSS_BETA)
            * tf.math.pow(y_pred_class, self.params.FOCAL_LOSS_ALPHA)
            * tf.math.log(tf.clip_by_value(1.0 - y_pred_class, 1e-4, 1. - 1e-4))
        )

        n = tf.reduce_sum(pos_mask)
        pos_loss_val = tf.reduce_sum(pos_loss)
        neg_loss_val = tf.reduce_sum(neg_loss)

        loss_val = tf.cond(tf.greater(n, 0), lambda: (pos_loss_val + neg_loss_val) / n, lambda: neg_loss_val)
        return loss_val

    def r_offset_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.r_offset_pos[0]:self.r_offset_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.r_offset_pos[0]:self.r_offset_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat, loss_type="mae")
        return loss_val

    def fullbox_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.fullbox_pos[0]:self.fullbox_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.fullbox_pos[0]:self.fullbox_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat, loss_type="mae")
        return loss_val

    def l_shape_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.l_shape_pos[0]:self.l_shape_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.l_shape_pos[0]:self.l_shape_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val
    
    def radial_dist_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.radial_dist_pos:self.radial_dist_pos + 1]
        y_pred_feat = y_pred[:, :, :, self.radial_dist_pos:self.radial_dist_pos + 1]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat, loss_type="mape")
        return loss_val

    def orientation_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.orientation_pos:self.orientation_pos + 1]
        y_pred_feat = y_pred[:, :, :, self.orientation_pos:self.orientation_pos + 1]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat, loss_type="mae")
        # sqrt(1-0.99*cos(2*x))+abs(x*x*0.05)-0.0999
        # This will have high loss at multiples of 90 deg, 0 loss at delta 0 and reduced loss at multiples of 180 deg
        loss_val = tf.math.sqrt(1.0 - (0.99 * tf.math.cos(2.0 * loss_val))) + tf.math.abs(loss_val * loss_val * 0.05) - 0.0999
        return loss_val

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

        total_loss = self.class_focal_loss(y_true, y_pred)

        if self.params.REGRESSION_FIELDS["r_offset"].active:
            total_loss += self.r_offset_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["r_offset"].loss_weight
        if self.params.REGRESSION_FIELDS["fullbox"].active:
            total_loss += self.fullbox_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["fullbox"].loss_weight
        if self.params.REGRESSION_FIELDS["l_shape"].active:
            total_loss += self.l_shape_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["l_shape"].loss_weight
        if self.params.REGRESSION_FIELDS["3d_info"].active:
            total_loss += self.radial_dist_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["3d_info"].loss_weight[0]
            total_loss += self.orientation_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["3d_info"].loss_weight[1]
            total_loss += self.obj_dims_loss(y_true, y_pred) * self.params.REGRESSION_FIELDS["3d_info"].loss_weight[2]

        return total_loss
