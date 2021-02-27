import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.losses import Loss, categorical_crossentropy
import pygame
from numba.typed import List
from common.utils import to_3channel
from data.label_spec import SEMSEG_CLASS_MAPPING


class SemsegLoss():
    def __init__(self, save_path=None):
        self.display = pygame.display.set_mode((640, 256*3), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.save_path = save_path
        self.step_counter = 0
        self.metrics = {
            "focal_tversky": tf.keras.metrics.Mean("focal_tversky")
        }

    def _show_semseg(self, inp, y_true, y_pred):
        inp_img = cv2.cvtColor(inp[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        semseg_true = cv2.cvtColor(to_3channel(y_true[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self.display.blit(surface_y_true, (0, 256))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, 256*2))

        if self.step_counter % 500 == 0:
            pygame.image.save(self.display, f"{self.save_path}/train_result_{self.step_counter}.png")

        pygame.display.flip()

    @staticmethod
    def tversky(y_true, y_pred, pos_mask):
        smooth = 1.0
        y_true_masked = y_true * pos_mask
        y_pred_masked = y_pred * pos_mask
        y_true_pos = tf.reshape(y_true_masked, [-1])
        y_pred_pos = tf.reshape(y_pred_masked, [-1])
        true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
        false_neg = tf.reduce_sum(y_true_pos * (1.0 - y_pred_pos))
        false_pos = tf.reduce_sum((1.0 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    @staticmethod
    def focal_tversky(y_true, y_pred, pos_mask):
        pt_1 = SemsegLoss.tversky(y_true, y_pred, pos_mask)
        gamma = 0.75
        return tf.math.pow((1.0 - pt_1), gamma)

    def calc(self, inp, y_true, pos_mask, y_pred):
        loss_sum = 0
        pos_mask_stacked = tf.stack([pos_mask] * y_true.shape[-1], axis=-1)

        focal = self.focal_tversky(y_true, y_pred, pos_mask_stacked)
        loss_sum += focal
        self.metrics["focal_tversky"].update_state(focal)

        self._show_semseg(inp, y_true, y_pred)
        self.step_counter += 1

        return loss_sum
