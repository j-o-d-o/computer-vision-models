import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.depth.params import Params
import matplotlib.pyplot as plt
import pygame
import numpy as np
from tensorflow.python.keras.utils import losses_utils


class DepthLoss(Loss):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None, save_path=None):
        super().__init__(reduction=reduction, name=name)
        self.display = pygame.display.set_mode((640*2, 256), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.save_path = save_path
        self.step_counter = 0

    def _cmap(self, depth_map):
        npumpy_depth_map = depth_map.numpy()
        npumpy_depth_map /= 140.0
        npumpy_depth_map *= 255.0
        npumpy_depth_map = np.clip(npumpy_depth_map, 0, 255)
        depth_stack = [npumpy_depth_map.astype(np.uint8).swapaxes(0, 1)]
        npumpy_depth_map = np.stack(depth_stack * 3, axis=-1)
        return npumpy_depth_map

    def _show_depthmaps(self, y_pred, y_true):
        surface_y_true = pygame.surfarray.make_surface(self._cmap(y_true[0]))
        self.display.blit(surface_y_true, (0, 0))
        surface_y_pred = pygame.surfarray.make_surface(self._cmap(y_pred[0]))
        self.display.blit(surface_y_pred, (640, 0))

        if self.step_counter % 500 == 0:
            pygame.image.save(self.display, f"{self.save_path}/train_result_{self.step_counter}.png")

        pygame.display.flip()

    @staticmethod
    def _calc_loss(y_true, y_pred):
        pos_mask = tf.cast(tf.greater(y_true, 1.0), tf.float32)
        n = tf.reduce_sum(pos_mask)
        # Point-wise depth
        # l_depth = pos_mask * tf.abs(y_pred - y_true)
        # l_depth = tf.reduce_sum(l_depth) / n

        loss_mask = pos_mask * ((y_true - y_pred) / tf.maximum(tf.math.abs(y_true), 1e-10))
        loss_mask = tf.math.abs(loss_mask)
        loss_val = tf.reduce_sum(loss_mask)
        loss_val = tf.cond(tf.greater(n, 0), lambda: loss_val / n, lambda: loss_val)
        return loss_val

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis=-1)

        self._show_depthmaps(y_pred, y_true)
        self.step_counter += 1

        return self._calc_loss(y_true, y_pred)
