import numpy as np
import pytest
from models.centernet import CenternetLoss
import tensorflow as tf


class TestLoss():
    def setup_method(self):
        # Object data
        self.cp_x = 1 
        self.cp_y = 1
        self.cls_idx = 1
        self.obj_data = {
            "loc_off": [0.2, 0.3], "width_px": 2.2, "height_px": 1.1,
            "bottom_left_off": [-2.0, 1.5], "bottom_right_off": [2.1, 1.2], "bottom_center_off": [0.5, 1.7], "center_height": 1.6,
            "radial_dist": 23.0, "orientation": 0.12, "obj_width": 1.8, "obj_height": 1.1, "obj_length": 2.9
        }

        # Create ground truth input
        self.mask_height = 7
        self.mask_width = 7
        self.nb_cls = 3
        self.channels = self.nb_cls + 16
        self.ground_truth = np.zeros((self.mask_height, self.mask_width, self.channels))
        # class with a bit of distribution to the right keypoint
        self.ground_truth[self.cp_y    ][self.cp_x][self.cls_idx] = 1.0
        self.ground_truth[self.cp_y + 1][self.cp_x][self.cls_idx] = 0.8
        # regression params
        self.ground_truth[self.cp_y][self.cp_x][self.nb_cls:] = [
            *self.obj_data["loc_off"], self.obj_data["width_px"], self.obj_data["height_px"],
            *self.obj_data["bottom_left_off"], *self.obj_data["bottom_right_off"], *self.obj_data["bottom_center_off"], self.obj_data["center_height"],
            self.obj_data["radial_dist"], self.obj_data["orientation"], self.obj_data["obj_width"], self.obj_data["obj_height"], self.obj_data["obj_length"]
        ]

        # Create perfect prediction
        self.perfect_prediction = self.ground_truth.copy()
        self.perfect_prediction[self.cp_y + 1][self.cp_x][self.cls_idx] = 0.0

        # Create loss class
        self.loss = CenternetLoss(self.nb_cls)

    def test_no_loss(self):
        no_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([self.perfect_prediction])).numpy()
        assert no_loss < 0.00001

    def test_class_loss(self):
        # Test class loss: peak not quite at 1.0
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y][self.cp_x][self.cls_idx] = 0.8
        class_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert class_loss < 0.01
        # Test class loss: one off peak vs random wrong peak
        prediction = self.perfect_prediction.copy()
        prediction[self.cp_y + 1][self.cp_x][self.cls_idx] = 1.0
        one_off_class_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        prediction[self.cp_y + 1][self.cp_x][self.cls_idx] = 0.0 # reset previous peak
        prediction[self.cp_y + 3][self.cp_x][self.cls_idx] = 1.0
        wrong_peak_class_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert one_off_class_loss < wrong_peak_class_loss

    def test_size_loss(self):
        # Test size loss: width
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        width_idx = self.nb_cls + 2
        height_idx = self.nb_cls + 3
        fp[width_idx] = self.obj_data["width_px"] + 10
        small_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        fp[width_idx] = self.obj_data["width_px"] - 30
        large_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert large_width_loss > small_width_loss > 0
        # Test width method directly
        width_loss = self.loss.size_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert width_loss > 0.0

        # Test size loss: height
        fp[width_idx] = self.obj_data["width_px"] # reset width to ground truth
        fp[height_idx] = self.obj_data["height_px"] + 10
        small_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        fp[height_idx] = self.obj_data["height_px"] - 30
        large_width_loss = self.loss(np.asarray([self.ground_truth]), np.asarray([prediction])).numpy()
        assert large_width_loss > small_width_loss > 0
        # Test height method directly
        height_loss = self.loss.size_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert height_loss > 0.0

    def test_bottom_edge_pts_loss(self):
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 4] = self.obj_data["bottom_left_off"][0] + 1
        fp[self.nb_cls + 5] = self.obj_data["bottom_left_off"][1] + 2
        loss_val1 = self.loss.bottom_edge_pts_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        fp[self.nb_cls + 6] = self.obj_data["bottom_right_off"][0] + 1
        fp[self.nb_cls + 7] = self.obj_data["bottom_right_off"][1] + 2
        loss_val2 = self.loss.bottom_edge_pts_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val1 < loss_val2

    def test_bottom_center_off_loss(self):
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 8] = self.obj_data["bottom_center_off"][0] + 1.5
        fp[self.nb_cls + 9] = self.obj_data["bottom_center_off"][1] + 1.5
        loss_val = self.loss.bottom_center_off_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.01

    def test_center_height_loss(self):
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 10] = self.obj_data["center_height"] + 3
        loss_val = self.loss.center_height_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.01

    def test_radial_dist_loss(self):
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 11] = self.obj_data["radial_dist"] + 2.5
        loss_val = self.loss.radial_dist_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.01

    def test_orientation_loss(self):
        # TODO: Make this one nicer, high loss at 90 deg error, smaller loss at 180
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 12] = self.obj_data["orientation"] + 1.5 # 90 deg error
        loss_val = self.loss.orientation_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.01

    def test_obj_dims_loss(self):
        # TODO: Make this one nicer, high loss at 90 deg error, smaller loss at 180
        prediction = self.perfect_prediction.copy()
        fp = prediction[self.cp_y][self.cp_x]
        fp[self.nb_cls + 13] = self.obj_data["obj_width"] + 0.5
        fp[self.nb_cls + 14] = self.obj_data["obj_height"] + 0.5
        fp[self.nb_cls + 15] = self.obj_data["obj_length"] + 0.5
        loss_val = self.loss.obj_dims_loss(tf.cast(np.asarray([self.ground_truth]), tf.float32), tf.cast(np.asarray([prediction]), tf.float32))
        assert loss_val > 0.01