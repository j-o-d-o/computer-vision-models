import numpy as np
import pytest
from models.centernet_2d import Centernet2DLoss


class TestLoss():
    def test_centernet2d_loss(self):
        # Mask data
        mask_height = 8
        mask_width = 8
        nb_classes = 3
        channels = nb_classes + 2 + 2

        # Object data
        keypoint_x = 1
        keypoint_y = 1
        obj_width = 2.2
        obj_height = 1.1
        cls_idx = 1

        # Create ground truth input
        ground_truth = np.zeros((mask_height, mask_width, channels))
        ground_truth[keypoint_y][keypoint_x][nb_classes + 2] = obj_width
        ground_truth[keypoint_y][keypoint_x][nb_classes + 3] = obj_height
        ground_truth[keypoint_y][keypoint_x][cls_idx] = 1.0
        ground_truth[keypoint_y + 1][keypoint_x][cls_idx] = 0.5
        ground_truth[keypoint_y - 1][keypoint_x][cls_idx] = 0.5
        ground_truth[keypoint_y][keypoint_x + 1][cls_idx] = 0.5
        ground_truth[keypoint_y][keypoint_x - 1][cls_idx] = 0.5


        # Create perfect prediction
        perfect_prediction = ground_truth.copy()
        perfect_prediction[keypoint_y + 1][keypoint_x][cls_idx] = 0.0
        perfect_prediction[keypoint_y - 1][keypoint_x][cls_idx] = 0.0
        perfect_prediction[keypoint_y][keypoint_x + 1][cls_idx] = 0.0
        perfect_prediction[keypoint_y][keypoint_x - 1][cls_idx] = 0.0

        centernet2dLoss = Centernet2DLoss(nb_classes, size_weight = 0.4, offset_weight = 0.2, focal_loss_alpha = 2.0, focal_loss_beta = 0.4)
        no_loss = centernet2dLoss(np.asarray([ground_truth]), np.asarray([perfect_prediction])).numpy()
        assert pytest.approx(no_loss) == 0.0


        # Create flawed size prediction
        prediction = perfect_prediction.copy()
        prediction[keypoint_y][keypoint_x][nb_classes + 2] = obj_width + 0.2
        prediction[keypoint_y][keypoint_x][nb_classes + 3] = obj_height - 0.2

        flawd_size_loss = centernet2dLoss(np.asarray([ground_truth]), np.asarray([prediction])).numpy()
        assert pytest.approx(flawd_size_loss, 0.01) == 0.16


        # Create flawed class
        prediction = perfect_prediction.copy()
        prediction[keypoint_y][keypoint_x][cls_idx] = 0.8
        prediction[keypoint_y][keypoint_x][cls_idx - 1] = 0.4
        prediction[keypoint_y][keypoint_x + 1][cls_idx] = 0.6

        flawd_class_loss = centernet2dLoss(np.asarray([ground_truth]), np.asarray([prediction])).numpy()
        assert pytest.approx(flawd_class_loss, 0.01) == 0.34

        # Only size predicted good
        only_size_predicted = np.zeros((mask_height, mask_width, channels))
        only_size_predicted[keypoint_y][keypoint_x][nb_classes + 2] = obj_width
        only_size_predicted[keypoint_y][keypoint_x][nb_classes + 3] = obj_height
        only_size_loss = centernet2dLoss(np.asarray([ground_truth]), np.asarray([only_size_predicted])).numpy()
        print(only_size_loss)

        # Create all zero
        no_detections = np.zeros((mask_height, mask_width, channels))
        no_detections_loss = centernet2dLoss(np.asarray([ground_truth]), np.asarray([no_detections])).numpy()
        print(no_detections_loss)

