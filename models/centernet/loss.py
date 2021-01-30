import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.centernet.params import Params


class CenternetLoss(Loss):
    def __init__(self, nb_cls,
        focal_loss_alpha = Params.FOCAL_LOSS_ALPHA,
        focal_loss_beta = Params.FOCAL_LOSS_ALPHA,
        cls_weight = Params.CLS_WEIGHT,
        offset_weight = Params.OFFSET_WEIGHT,
        size_weight = Params.SIZE_WEIGHT,
        box3d_weight = Params.BOX3D_WEIGHT,
        radial_dist_weight = Params.RADIAL_DIST_WEIGHT,
        orientation_weight = Params.ORIENTATION_WEIGHT,
        obj_dims_weight = Params.OBJ_DIMS_WEIGHT
    ):
        super().__init__()
        # hyperparams
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_beta = focal_loss_beta
        self.cls_weight = cls_weight
        self.offset_weight = offset_weight ** 2
        self.size_weight = size_weight ** 2
        self.box3d_weight = box3d_weight ** 2
        self.radial_dist_weight = radial_dist_weight
        self.orientation_weight = orientation_weight ** 2
        self.obj_dims_weight = obj_dims_weight ** 2
        # position of data within the y_true and y_pred
        self.class_array_pos       = [0          , nb_cls     ] # object classes
        self.loc_offset_pos        = [nb_cls     , nb_cls +  2] # x and y location offset resulting from scaling with R
        self.size_px_pos           = [nb_cls +  2, nb_cls +  4] # width and height in pixel (fullbox)
        self.bottom_edge_pts_pos   = [nb_cls +  4, nb_cls +  8] # bottom_left_off, bottom_right_off
        self.bottom_center_off_pos = [nb_cls +  8, nb_cls + 10] # bottom_center_off (has some cases where loss is reduced)
        self.center_height_pos     = nb_cls + 10
        self.radial_dist_pos       = nb_cls + 11
        self.orientation_pos       = nb_cls + 12
        self.obj_dims_pos          = [nb_cls + 13, nb_cls + 16] # width, height, length in meter

    def class_focal_loss(self, y_true, y_pred):
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
        y_true_feat = y_true[:, :, :, self.size_px_pos[0]:self.size_px_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.size_px_pos[0]:self.size_px_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def loc_offset_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.loc_offset_pos[0]:self.loc_offset_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.loc_offset_pos[0]:self.loc_offset_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def bottom_edge_pts_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.bottom_edge_pts_pos[0]:self.bottom_edge_pts_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.bottom_edge_pts_pos[0]:self.bottom_edge_pts_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def bottom_center_off_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.bottom_center_off_pos[0]:self.bottom_center_off_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.bottom_center_off_pos[0]:self.bottom_center_off_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def center_height_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.center_height_pos:self.center_height_pos + 1]
        y_pred_feat = y_pred[:, :, :, self.center_height_pos:self.center_height_pos + 1]
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
        # This will have high loss at multiples of 90 deg, 0 loss at delta 0 and
        # very reduced loss at multiples of 180 deg
        loss_val = tf.math.sqrt(1.0 - (0.99 * tf.math.cos(2.0 * loss_val))) + tf.math.abs(loss_val * loss_val * 0.05) - 0.0999
        return loss_val

    def obj_dims_loss(self, y_true, y_pred):
        y_true_feat = y_true[:, :, :, self.obj_dims_pos[0]:self.obj_dims_pos[1]]
        y_pred_feat = y_pred[:, :, :, self.obj_dims_pos[0]:self.obj_dims_pos[1]]
        loss_val = self.calc_loss(y_true, y_true_feat, y_pred_feat)
        return loss_val

    def calc_loss(self, y_true, y_true_feat, y_pred_feat, loss_type: str = "mse"):
        y_true_class = y_true[:, :, :, :self.class_array_pos[1]]

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

        class_loss = self.class_focal_loss(y_true, y_pred)
        offset_loss = self.loc_offset_loss(y_true, y_pred)
        size_loss = self.size_loss(y_true, y_pred)
        bottom_edge_pts_loss = self.bottom_edge_pts_loss(y_true, y_pred)
        bottom_center_off_loss = self.bottom_center_off_loss(y_true, y_pred)
        center_height_loss = self.center_height_loss(y_true, y_pred)
        radial_dist_loss = self.radial_dist_loss(y_true, y_pred)
        orientation_loss = self.orientation_loss(y_true, y_pred)
        obj_dims_loss = self.obj_dims_loss(y_true, y_pred)

        total_loss = (self.cls_weight * class_loss) + (self.offset_weight * offset_loss) + (self.size_weight * size_loss) + \
            (self.box3d_weight * bottom_edge_pts_loss) + (self.box3d_weight * bottom_center_off_loss) + (self.box3d_weight * center_height_loss) + \
            (self.radial_dist_weight * radial_dist_loss) + (self.orientation_weight * orientation_loss) + (self.obj_dims_weight * obj_dims_loss)

        return total_loss
