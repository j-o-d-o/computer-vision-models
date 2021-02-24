import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.losses import Loss, categorical_crossentropy
import pygame
from common.utils import cmap_depth, to_3channel
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.semseg.loss import SemsegLoss
from models.depth.loss import DepthLoss


class MultitaskLoss():
    def __init__(self):
        self.metrics = {
            "semseg_focal_tversky": tf.keras.metrics.Mean("semseg_focal_tversky"),
            "depth_mape": tf.keras.metrics.Mean("depth_mape")
        }

    def calc_depth(self, y_true, y_pred):
        loss = DepthLoss.calc(y_true, y_pred)
        self.metrics["depth_mape"].update_state(loss)
        return loss

    def calc_semseg(self, y_true, pos_mask, y_pred):
        pos_mask_stacked = tf.stack([pos_mask] * y_true.shape[-1], axis=-1)
        pos_mask_count = tf.reduce_sum(pos_mask)
        loss = SemsegLoss.focal_tversky(y_true, y_pred, pos_mask_stacked)
        self.metrics["semseg_focal_tversky"].update_state(loss)
        return loss
