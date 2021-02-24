import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.losses import Loss, categorical_crossentropy
import pygame
from common.utils import cmap_depth, to_3channel
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.dmds_ref.regularizers import joint_bilateral_smoothing


class SemsegLoss():
    def __init__(self, save_path=None):
        self.display = pygame.display.set_mode((640, 256*3), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.save_path = save_path
        self.step_counter = 0
        self.metrics = {
            "ce": tf.keras.metrics.Mean("cd"),
            "focal_tversky": tf.keras.metrics.Mean("focal_tversky")
        }

    def _show_semseg(self, inp, y_true, y_pred):
        inp_img = cv2.cvtColor(inp[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        semseg_true = cv2.cvtColor(to_3channel(y_true[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_true = pygame.surfarray.make_surface(semseg_true)
        self.display.blit(surface_y_true, (0, 256))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, 256*2))

        if self.step_counter % 500 == 0:
            pygame.image.save(self.display, f"{self.save_path}/train_result_{self.step_counter}.png")

        pygame.display.flip()

    def class_focal_loss(self, y_true, y_pred):
        pos_loss = (
            -tf.math.pow(1.0 - y_true, 2.0)
            * tf.math.log(tf.clip_by_value(y_true, 0.01, 0.99))
        )
        neg_loss = (
            -tf.math.pow(1.0 - y_true, 4.0)
            * tf.math.pow(y_pred, 2.0)
            * tf.math.log(tf.clip_by_value(1.0 - y_pred, 0.01, 0.99))
        )

        pos_loss_val = tf.reduce_mean(pos_loss)
        neg_loss_val = tf.reduce_mean(neg_loss)

        return pos_loss_val + neg_loss_val

    def tversky(self, y_true, y_pred, pos_mask):
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

    def focal_tversky(self, y_true,y_pred, pos_mask):
        pt_1 = self.tversky(y_true, y_pred, pos_mask)
        gamma = 0.75
        return tf.math.pow((1.0 - pt_1), gamma)

    def calc(self, inp, y_true, pos_mask, y_pred):
        loss_sum = 0
        pos_mask_stacked = tf.stack([pos_mask] * y_true.shape[-1], axis=-1)
        pos_mask_count = tf.reduce_sum(pos_mask)

        # ce = tf.reduce_sum(categorical_crossentropy(y_true, y_pred, from_logits=True) * pos_mask) / pos_mask_count
        # loss_sum += ce
        # self.metrics["ce"].update_state(ce)

        focal = self.focal_tversky(y_true, y_pred, pos_mask_stacked)
        loss_sum += focal
        self.metrics["focal_tversky"].update_state(focal)

        self._show_semseg(inp, y_true, y_pred)
        self.step_counter += 1

        return loss_sum
