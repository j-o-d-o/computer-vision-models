from ssd.src.loss import SSDLoss
import numpy as np
import tensorflow as tf


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        :return: None
        """
        self.num_classes = 3
        self.num_boxes = 3
        self.ground_truth = np.array([
            [
                0.0, 0.0, 1.0, 0.1, 1.0, -0.2, -0.3, # box 1
                0.0, 1.0, 0.0, 0.1, 0.2, 0.1, 0.3, # box 2
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # box 3
            ], # batch 1
            [
                1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, # box 1
                1.0, 0.0, 0.0, 0.2, 0.4, 0.3, 0.1, # box 2
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # box 3
            ], # batch 2
        ], np.float32)
        self.prediction = np.array([
            [
                0.0, 10.0, -2.0, 0.1, 1.2, 0.2, 0.1, # box 1
                0.3, 0.0, -0.1, 0.2, 0.2, 0.1, 0.3, # box 2
                0.0, 0.1, 0.0, 4.0, 4.0, 4.0, 4.0, # box 3
            ], # batch 1
            [
                0.0, 0.7, 0.1, 0.0, 0.1, 0.1, 0.0, # box 1
                0.0, 0.3, 0.2, 0.3, 0.4, 0.2, 0.1, # box 2
                0.0, 0.1, 0.0, 4.0, 4.0, 4.0, 4.0, # box 3
            ], # batch 2
        ], np.float32)

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def test_ssd_loss(self):
        ssd_loss = SSDLoss(self.num_boxes, self.num_classes, 1)
        ssd_loss.compute_loss(self.ground_truth, self.prediction)
