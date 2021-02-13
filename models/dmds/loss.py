import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams
import numpy as np
from pymongo import MongoClient
import cv2
from common.utils import resize_img
import matplotlib.pyplot as plt
# from models.dmds.resampler import resampler_with_unstacked_warp
from models.dmds.ref_imp import consistency_losses
from models.dmds.ref_imp import transform_depth_map


class DmdsLoss:
    def __init__(self):
        self.loss_vals = {
            # "depth_abs": None,
            "depth_smooth": None,
            "depth_var": None,
            "mm_sparsity": None,
            "mm_smooth": None,
            "depth_error": None,
            "rgb_error": None,
            "ssim_error": None,
            "rotation_error": None,
            "translation_error": None,
        }
    
    def norm(self, x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    def expand_dims_twice(self, x, dim):
        return tf.expand_dims(tf.expand_dims(x, dim), dim)

    def normalize_motion_map(self, res_motion_map, motion_map):
        """Normalizes a residual motion map by the motion map's norm."""
        norm = tf.reduce_mean(tf.square(motion_map), axis=[1, 2, 3], keepdims=True) * 3.0
        return res_motion_map * tf.math.reciprocal_no_nan(tf.sqrt(norm + 1e-12))

    def calc_l12_sparsity(self, motion_map):
        tensor_abs = tf.abs(motion_map)
        mean = tf.stop_gradient(tf.reduce_mean(tensor_abs, axis=[1, 2], keepdims=True))
        self.loss_vals["mm_sparsity"] = tf.reduce_mean(2 * mean * tf.sqrt(tensor_abs * tf.math.reciprocal_no_nan(mean + 1e-24) + 1)) * 1.0

    def calc_l1_smoothness(self, mm):
        grad_dx = mm - tf.roll(mm, 1, 1)
        grad_dy = mm - tf.roll(mm, 1, 2)
        grad_dx = grad_dx[:, 1:, 1:, :]
        grad_dy = grad_dy[:, 1:, 1:, :]
        self.loss_vals["mm_smooth"] = tf.reduce_mean(tf.sqrt(1e-24 + tf.square(grad_dx) + tf.square(grad_dy))) * 1.0

    def calc_depth_smoothness(self, depth_stack, rgb_stack):
        def _gradient_x(img):
            return img[:, :, :-1, :] - img[:, :, 1:, :]
        def _gradient_y(img):
            return img[:, :-1, :, :] - img[:, 1:, :, :]

        mean_depth = tf.reduce_mean(depth_stack)
        depth_var = tf.reduce_mean(tf.square(depth_stack * tf.math.reciprocal_no_nan(mean_depth) - 1.0))
        self.loss_vals["depth_var"] = tf.math.reciprocal_no_nan(depth_var) * 0.001

        disp = tf.math.reciprocal_no_nan(depth_stack)
        mean_disp = tf.reduce_mean(disp, axis=[1, 2, 3], keepdims=True)
        disp = disp * tf.math.reciprocal_no_nan(mean_disp)

        disp_dx = _gradient_x(disp)
        disp_dy = _gradient_y(disp)
        img_dx = _gradient_x(rgb_stack)
        img_dy = _gradient_y(rgb_stack)
        weights_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keepdims=True))
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        self.loss_vals["depth_smooth"] = (tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))) * 0.01

    # def calc_depth_mask(self, x, gt):
    #     # All 0 values should not contribute, but network should not learn this mask
    #     pos_mask = tf.stop_gradient(tf.cast(tf.greater(gt, 0.1), tf.float32))
    #     n = tf.reduce_sum(pos_mask)
    #     loss_val = 0
    #     loss_val = pos_mask * (x - gt)
    #     loss_val = tf.math.abs(loss_val)
    #     loss_val = tf.reduce_sum(loss_val)
    #     loss_val = tf.cond(tf.greater(n, 0), lambda: (loss_val) / n, lambda: loss_val)
    #     self.loss_vals["depth_abs"] = loss_val * 0.025

    def calc(self, img0, img1, x0, x1, mm, mm_inv, tran, tran_inv, rot, rot_inv, intr, mask0, mask1):
        # softplus activation for depth maps
        # x0 = tf.math.log(1.0 + tf.math.exp(x0))
        # x1 = tf.math.log(1.0 + tf.math.exp(x1))

        # Depthmap regualizer
        # ==========================================================================
        rgb_stack = tf.concat([img0, img1], axis=0)
        depth_stack = tf.concat([x0, x1], axis=0)
        self.calc_depth_smoothness(depth_stack, rgb_stack)

        # Depthmap learning
        # ==========================================================================
        # mask_stack = tf.concat([mask0, mask1], axis=0)
        # self.calc_depth_mask(tf.squeeze(depth_stack, axis=3), mask_stack)

        # Motionmap regualizer
        # ==========================================================================
        background_translation = tf.broadcast_to(self.expand_dims_twice(tran, -2), shape=tf.shape(mm))
        background_translation_inv = tf.broadcast_to(self.expand_dims_twice(tran, -2), shape=tf.shape(mm))
        background_translation_stack = tf.concat([background_translation, background_translation_inv], axis=0)
        residual_translation_stack = tf.concat([mm, mm_inv], axis=0)
        translation_stack = residual_translation_stack + background_translation_stack
        normalized_trans = self.normalize_motion_map(residual_translation_stack, translation_stack)
        self.calc_l1_smoothness(normalized_trans)
        self.calc_l12_sparsity(normalized_trans)

        # Cosistency RGB Depth Translation Rotation
        # ==========================================================================
        rotation_stack = tf.concat([rot, rot_inv], axis=0)
        intr_stack = tf.concat([intr, intr], axis=0)
        intr_inv_stack = tf.linalg.inv(intr_stack)
        transformed_depth = transform_depth_map.using_motion_vector(tf.squeeze(depth_stack, axis=-1),
            translation_stack, rotation_stack, intr_stack, intr_inv_stack)

        flipped_rgb_stack = tf.concat([img1, img0], axis=0)
        flipped_depth_stack = tf.concat([x1, x0], axis=0)
        flipped_depth_stack = tf.stop_gradient(flipped_depth_stack)
        flipped_rotation_stack = tf.concat([rot_inv, rot], axis=0)
        filpped_background_translation_stack = tf.concat([background_translation_inv, background_translation], axis=0)
        flipped_residual_translation_stack = tf.concat([mm_inv, mm], axis=0)
        flipped_translation_stack = flipped_residual_translation_stack + filpped_background_translation_stack

        loss_endpoints = consistency_losses.rgbd_and_motion_consistency_loss(
            transformed_depth, rgb_stack, flipped_depth_stack, flipped_rgb_stack,
            rotation_stack, translation_stack, flipped_rotation_stack, flipped_translation_stack, None)

        self.loss_vals["depth_error"] = loss_endpoints["depth_error"] * 0.01 # 1.0e-5
        self.loss_vals["rgb_error"] = loss_endpoints["rgb_error_mean"] * 2.0
        self.loss_vals["ssim_error"] = loss_endpoints["ssim_error_mean"] * 4.0
        self.loss_vals["rotation_error"] = loss_endpoints["rotation_error"] * 0.05 # 1.0e-3
        self.loss_vals["translation_error"] = loss_endpoints["translation_error"] * 0.1 # 5.0e-2

        loss_val = self.loss_vals["depth_smooth"] + self.loss_vals["depth_var"] + self.loss_vals["mm_sparsity"] + \
            self.loss_vals["mm_smooth"] + self.loss_vals["depth_error"] + self.loss_vals["rgb_error"] + \
            self.loss_vals["ssim_error"] + self.loss_vals["rotation_error"] + self.loss_vals["translation_error"]

        return loss_val

def test():
    loss = DmdsLoss()
    batch_size = 8

    client = MongoClient("mongodb://localhost:27017")
    collection = client["depth"]["driving_stereo"]
    documents = collection.find({}).limit(10)
    documents = list(documents)
    for i in range(0, len(documents)-1):
        intr = np.array([
            [375.0,  0.0, 160.0],
            [ 0.0, 375.0, 128.0],
            [ 0.0,   0.0,   1.0]
        ], dtype=np.float32)

        img0 = cv2.imdecode(np.frombuffer(documents[i]["img"], np.uint8), cv2.IMREAD_COLOR)
        img0, _ = resize_img(img0, 320, 128, 0)

        img1 = cv2.imdecode(np.frombuffer(documents[i+1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img1, _ = resize_img(img1, 320, 128, 0)

        # create gt depth_maps
        x0 = cv2.imdecode(np.frombuffer(documents[i]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x0, _ = resize_img(x0, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x0 = np.expand_dims(x0, axis=-1)

        x1 = cv2.imdecode(np.frombuffer(documents[i+1]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x1, _ = resize_img(x1, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x1 = np.expand_dims(x1, axis=-1)

        mm = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm[20:100, 20:100, :] = [2.0, 0.0, 0.0]

        rot = np.zeros((batch_size, 3), dtype=np.float32)
        rot[:,] = np.array([0.0, 0.1, 0.0])
        tran = np.zeros((batch_size, 3), dtype=np.float32)
        tran[:,] = np.array([0.0, 0.0, 0.0])

        x0 = np.stack([x0]*batch_size, axis=0)
        x0 = x0.astype(np.float32)
        x1 = np.stack([x1]*batch_size, axis=0)
        x1 = x1.astype(np.float32)
        img0 = np.stack([img0]*batch_size, axis=0)
        img0 = img0.astype(np.float32)
        img1 = np.stack([img1]*batch_size, axis=0)
        img1 = img1.astype(np.float32)
        mm = np.stack([mm]*batch_size, axis=0)
        intr = np.stack([intr]*batch_size)

        x0 /= 256.0
        x1 /= 256.0
        loss.calc(img0, img1, x0, x1, mm, mm, tran, tran, rot, rot, intr)

if __name__ == "__main__":
    test()
