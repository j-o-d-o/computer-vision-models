import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams
import numpy as np
from pymongo import MongoClient
import cv2
import matplotlib.pyplot as plt
from common.utils import resize_img
# from models.dmds.resampler import resampler_with_unstacked_warp
from models.dmds_ref.resampler import resampler_with_unstacked_warp
from data.driving_stereo_depth import fill_depth_data


class DmdsLoss:
    def __init__(self):
        self.loss_vals = {
            "depth_abs": 0,
            "depth_smooth": 0,
            "depth_var": 0,
            "mm_sparsity": 0,
            "mm_smooth": 0,
            "depth": 0,
            "rgb": 0,
            "ssim": 0,
            "rot": 0,
            "tran": 0,
        }

        # some debug data
        self.resampled_img1 = None
        self.resampled_img1_masked = None
        self.resampled_depth1 = None
        self.resampled_depth1_masked = None
        self.warp_mask = None
        self.frame_closer_to_cam_mask = None

    def norm(self, x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    def expand_dims_twice(self, x, dim):
        return tf.expand_dims(tf.expand_dims(x, dim), dim)

    def construct_rotation_matrix(self, rot):
        sin_angles = tf.sin(rot)
        cos_angles = tf.cos(rot)
        # R = R_z * R_y * R_x
        sin_angles.shape.assert_is_compatible_with(cos_angles.shape)
        sx, sy, sz = tf.unstack(sin_angles, axis=-1)
        cx, cy, cz = tf.unstack(cos_angles, axis=-1)
        m00 = cy * cz
        m01 = (sx * sy * cz) - (cx * sz)
        m02 = (cx * sy * cz) + (sx * sz)
        m10 = cy * sz
        m11 = (sx * sy * sz) + (cx * cz)
        m12 = (cx * sy * sz) - (sx * cz)
        m20 = -sy
        m21 = sx * cy
        m22 = cx * cy
        matrix = tf.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), axis=-1)
        output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)

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
        self.loss_vals["depth_var"] = tf.math.reciprocal_no_nan(depth_var) * 0.1

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
        self.loss_vals["depth_smooth"] = (tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))) * 0.001

    def calc_depth_mask(self, y_pred, y_true):
        pos_mask = tf.cast(tf.greater(y_true, 1.0), tf.float32)
        n = tf.reduce_sum(pos_mask)
        y_true = y_true + 1e-10
        y_pred = y_pred + 1e-10
        disp_true = tf.math.reciprocal_no_nan(y_true)
        disp_pred = tf.math.reciprocal_no_nan(y_pred)
        loss_val = 0
        loss_val = pos_mask * (disp_pred - disp_true)
        loss_val = tf.math.abs(loss_val)
        loss_val = tf.reduce_sum(loss_val)
        loss_val = tf.cond(tf.greater(n, 0), lambda: (loss_val) / n, lambda: loss_val)
        self.loss_vals["depth_abs"] = loss_val * 3.5

    def warp_it(self, depth, translation, rotation, intrinsic_mat, intrinsic_mat_inv):
        _, height, width = tf.unstack(tf.shape(depth))
        grid = tf.squeeze(tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)
        grid = tf.cast(grid, tf.float32)

        cam_coords = tf.einsum('bij,jhw,bhw->bihw', intrinsic_mat_inv, grid, depth)
        xyz = (tf.einsum('bij,bjk,bkhw->bihw', intrinsic_mat, rotation, cam_coords) + tf.einsum('bij,bhwj->bihw', intrinsic_mat, translation))

        x, y, z = tf.unstack(xyz, axis=1)
        pixel_x = x / z
        pixel_y = y / z

        def _tensor(x):
            return tf.cast(tf.convert_to_tensor(x), tf.float32)

        x_not_underflow = pixel_x >= 0.0
        y_not_underflow = pixel_y >= 0.0
        x_not_overflow = pixel_x < _tensor(width - 1)
        y_not_overflow = pixel_y < _tensor(height - 1)
        z_positive = z > 0.0
        x_not_nan = tf.math.logical_not(tf.compat.v1.is_nan(pixel_x))
        y_not_nan = tf.math.logical_not(tf.compat.v1.is_nan(pixel_y))
        not_nan = tf.logical_and(x_not_nan, y_not_nan)
        not_nan_mask = tf.cast(not_nan, tf.float32)
        pixel_x = tf.math.multiply_no_nan(pixel_x, not_nan_mask)
        pixel_y = tf.math.multiply_no_nan(pixel_y, not_nan_mask)
        pixel_x = tf.clip_by_value(pixel_x, 0.0, _tensor(width - 1))
        pixel_y = tf.clip_by_value(pixel_y, 0.0, _tensor(height - 1))
        mask_stack = tf.stack([x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow, z_positive, not_nan], axis=0)
        mask = tf.reduce_all(mask_stack, axis=0)

        return pixel_x, pixel_y, z, mask

    def calc_warp_error(self, img0, img1, x1, px, py, z, mask):
        frame1_rgbd = tf.concat([img1, x1], axis=-1)
        frame1_rgbd_resampled = resampler_with_unstacked_warp(frame1_rgbd, px, py)
        img1_resampled, x1_resampled = tf.split(frame1_rgbd_resampled, [3, 1], axis=-1)
        x1_resampled = tf.squeeze(x1_resampled, axis=-1)

        mask = tf.stop_gradient(mask)
        frame0_closer_to_camera = tf.cast(tf.logical_and(mask, tf.less_equal(z, x1_resampled)), tf.float32)
        # frame0_closer_to_camera = tf.cast(mask, tf.float32)
        frame0_closer_to_camera_3c = tf.stack([frame0_closer_to_camera] * 3, axis=-1)
        depth_l1_diff = tf.abs(x1_resampled - z)
        depth_error = tf.reduce_mean(tf.math.multiply_no_nan(depth_l1_diff, frame0_closer_to_camera))

        rgb_l1_diff = tf.abs(img1_resampled - img0)
        rgb_error = tf.reduce_mean(tf.math.multiply_no_nan(rgb_l1_diff, tf.expand_dims(frame0_closer_to_camera, -1)))

        self.resampled_img1 = img1_resampled
        self.resampled_img1_masked = tf.math.multiply_no_nan(img1_resampled, frame0_closer_to_camera_3c)
        self.resampled_depth1 = tf.expand_dims(x1_resampled, -1)
        self.resampled_depth1_masked = tf.expand_dims(tf.math.multiply_no_nan(x1_resampled, frame0_closer_to_camera), -1)
        self.warp_mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        self.frame_closer_to_cam_mask = tf.expand_dims(frame0_closer_to_camera, -1)

        def weighted_average(x, w, epsilon=1.0):
            weighted_sum = tf.reduce_sum(x * w, axis=(1, 2), keepdims=True)
            sum_of_weights = tf.reduce_sum(w, axis=(1, 2), keepdims=True)
            return weighted_sum / (sum_of_weights + epsilon)

        def weighted_ssim(x, y, weight, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
            def _avg_pool3x3(x):
                return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')

            def weighted_avg_pool3x3(z):
                wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
                return wighted_avg * inverse_average_pooled_weight

            weight = tf.expand_dims(weight, -1)
            average_pooled_weight = _avg_pool3x3(weight)
            weight_plus_epsilon = weight + weight_epsilon
            inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

            mu_x = weighted_avg_pool3x3(x)
            mu_y = weighted_avg_pool3x3(y)
            sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
            sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
            sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
            if c1 == float('inf'):
                ssim_n = (2 * sigma_xy + c2)
                ssim_d = (sigma_x + sigma_y + c2)
            elif c2 == float('inf'):
                ssim_n = 2 * mu_x * mu_y + c1
                ssim_d = mu_x**2 + mu_y**2 + c1
            else:
                ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
                ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            result = ssim_n / ssim_d
            return tf.clip_by_value((1 - result) / 2, 0, 1), average_pooled_weight

        sec_moment = weighted_average(tf.square(x1_resampled - z), frame0_closer_to_camera) + 1e-4
        depth_proximity_weight = tf.math.multiply_no_nan(sec_moment * 
            tf.math.reciprocal_no_nan(tf.square(x1_resampled - z) + sec_moment), tf.cast(mask, tf.float32))
        depth_proximity_weight = tf.stop_gradient(depth_proximity_weight)

        ssim_error, avg_weight = weighted_ssim(img1_resampled, img0, depth_proximity_weight, c1=float('inf'), c2=9e-6)
        ssim_error_mean = tf.reduce_mean(tf.math.multiply_no_nan(ssim_error, avg_weight))
        
        # ssim_error = tf.image.ssim(
        #     tf.image.rgb_to_grayscale(tf.math.multiply_no_nan(img0, frame0_closer_to_camera_3c)),
        #     tf.image.rgb_to_grayscale(tf.math.multiply_no_nan(img1_resampled, frame0_closer_to_camera_3c)),
        #     255.0)
        # ssim_error_mean = (1 - tf.reduce_mean(ssim_error)) * 0.5

        self.loss_vals["rgb"] = 1.0 * rgb_error
        self.loss_vals["depth"] = 1.0e-4 * depth_error
        self.loss_vals["ssim"] = 3.0 * ssim_error_mean

        return frame0_closer_to_camera

    def calc_motion_field_consistency_loss(self, T, T_inv, R, R_inv, mask, px, py):
        R_unit = tf.matmul(R, R_inv)
        identity = tf.eye(3, batch_shape=tf.shape(R_unit)[:-2])
        R_error = tf.reduce_mean(tf.square(R_unit - identity), axis=(1, 2))
        rot1_scale = tf.reduce_mean(tf.square(R - identity), axis=(1, 2))
        rot2_scale = tf.reduce_mean(tf.square(R_inv - identity), axis=(1, 2))
        R_error /= (1e-24 + rot1_scale + rot2_scale)
        R_error = tf.reduce_mean(R_error)
        self.loss_vals["rot"] = R_error * 0.05

        T_inv_resampled = resampler_with_unstacked_warp(T_inv, tf.stop_gradient(px), tf.stop_gradient(py))
        
        rot_field_shape = tf.shape(tf.stack([T]*3, axis=-1)) # stupid way to get the correct shape for the matrix filed for the rotation...
        R_inv_field = tf.broadcast_to(self.expand_dims_twice(R_inv, -3), rot_field_shape)
        r2t1 = tf.matmul(R_inv_field, tf.expand_dims(T, -1))
        r2t1 = tf.squeeze(r2t1, axis=-1)
        trans_zero = r2t1 + T_inv_resampled

        t = tf.math.multiply_no_nan(mask, self.norm(trans_zero))
        # t = self.norm(trans_zero)

        translation_error = tf.reduce_mean(t / (1e-10 + self.norm(T) + self.norm(T_inv_resampled)))
        self.loss_vals["tran"] = translation_error * 0.1


    def calc(self, img0, img1, x0, x1, mm, mm_inv, tran, tran_inv, rot, rot_inv, intr, gt_x0, gt_x1):
        # softplus activation for depth maps
        # x0 = tf.math.log(1.0 + tf.math.exp(x0))
        # x1 = tf.math.log(1.0 + tf.math.exp(x1))

        # Data construction
        # ==========================================================================
        rgb_stack = tf.concat([img0, img1], axis=0)
        flipped_rgb_stack = tf.concat([img1, img0], axis=0)
        depth_stack = tf.concat([x0, x1], axis=0)
        flipped_depth_stack = tf.concat([x1, x0], axis=0)

        bg_translation = tf.broadcast_to(self.expand_dims_twice(tran, -2), shape=tf.shape(mm))
        bg_translation_inv = tf.broadcast_to(self.expand_dims_twice(tran_inv, -2), shape=tf.shape(mm))
        T = mm + bg_translation
        T_inv = mm_inv + bg_translation_inv
        T_stack = tf.concat([T, T_inv], axis=0)
        flipped_T_stack = tf.concat([T_inv, T], axis=0)

        R = self.construct_rotation_matrix(rot) # rot = [pitch, yaw, roll]
        R_inv = self.construct_rotation_matrix(rot_inv)
        R_stack = tf.concat([R, R_inv], axis=0)
        flipped_R_stack = tf.concat([R_inv, R], axis=0)

        K = intr
        K_inv = tf.linalg.inv(K)
        K_stack = tf.concat([K, K], axis=0)
        K_inv_stack = tf.concat([K_inv, K_inv], axis=0)

        # Depthmap regualizer
        # ==========================================================================
        self.calc_depth_smoothness(x0, img0)

        # Depthmap learning
        # ==========================================================================
        # self.calc_depth_mask(x0, gt_x0)

        # Motionmap regualizer
        # ==========================================================================
        mm_stack = tf.concat([mm, mm_inv], axis=0)
        normalized_trans = self.normalize_motion_map(mm_stack, T_stack)
        self.calc_l1_smoothness(normalized_trans)
        self.calc_l12_sparsity(normalized_trans)

        # Consistency Regularization
        # ==========================================================================
        # Warp img0 and x0 into frame t1
        px, py, z, mask = self.warp_it(tf.squeeze(depth_stack, axis=-1), T_stack, R_stack, K_stack, K_inv_stack)

        updated_mask = self.calc_warp_error(rgb_stack, flipped_rgb_stack, flipped_depth_stack, px, py, z, mask)
        self.calc_motion_field_consistency_loss(T_stack, flipped_T_stack, R_stack, flipped_R_stack, updated_mask, px, py)

        loss_val = self.loss_vals["depth_smooth"] + self.loss_vals["depth_var"] + self.loss_vals["mm_sparsity"] + \
            self.loss_vals["mm_smooth"] + self.loss_vals["depth"] + self.loss_vals["rgb"] + \
            self.loss_vals["ssim"] + self.loss_vals["rot"] + self.loss_vals["tran"] + self.loss_vals["depth_abs"]

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
        intr = np.stack([intr]*batch_size)

        img0 = cv2.imdecode(np.frombuffer(documents[i]["img"], np.uint8), cv2.IMREAD_COLOR)
        img0, _ = resize_img(img0, 320, 128, 0)
        img0 = np.stack([img0]*batch_size, axis=0)
        img0 = img0.astype(np.float32)

        img1 = cv2.imdecode(np.frombuffer(documents[i+1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img1, _ = resize_img(img1, 320, 128, 0)
        img1 = np.stack([img1]*batch_size, axis=0)
        img1 = img1.astype(np.float32)

        # create gt depth_maps
        x0 = cv2.imdecode(np.frombuffer(documents[i]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x0, _ = resize_img(x0, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x0 = fill_depth_data(x0)
        x0 = np.expand_dims(x0, axis=-1)
        x0 = np.stack([x0]*batch_size, axis=0)
        x0 = x0.astype(np.float32)
        x0 /= 255.0
        x0[:] = 0.01

        x1 = cv2.imdecode(np.frombuffer(documents[i+1]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x1, _ = resize_img(x1, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x1 = fill_depth_data(x1)
        x1 = np.expand_dims(x1, axis=-1)
        x1 = np.stack([x1]*batch_size, axis=0)
        x1 = x1.astype(np.float32)
        x1 /= 255.0
        x1[:] = 0.01

        mm = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm[:, 23:85, 132:320, :] = [0.0, 0.0, 0.0]

        mm_inv = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm_inv[:, 23:85, 172:320, :] = [0.0, 0.0, 0.0]

        rot = np.zeros((batch_size, 3), dtype=np.float32)
        rot[:,] = np.array([-0.0, 0.0, 0.0])
        rot_inv = np.zeros((batch_size, 3), dtype=np.float32)
        rot_inv[:,] = np.array([0.0, 0.0, 0.0])

        tran = np.zeros((batch_size, 3), dtype=np.float32)
        tran[:,] = np.array([0.0, 0.0, 0]) # [left,right | up,down | forward,backward]
        tran_inv = np.zeros((batch_size, 3), dtype=np.float32)
        tran_inv[:,] = np.array([0.0, 0.0, 0])

        loss.calc(img0, img1, x0, x1, mm, mm_inv, tran, tran_inv, rot, rot_inv, intr, x0, x1)

        for key in loss.loss_vals.keys():
            print(f"{key}: {loss.loss_vals[key].numpy()}")
        print(" - - - - - - - -")

        f, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42), (ax51, ax52)) = plt.subplots(5, 2)
        # img0, img1
        ax11.imshow(cv2.cvtColor(img0[0].astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax12.imshow(cv2.cvtColor(img1[0].astype(np.uint8), cv2.COLOR_BGR2RGB))
        # x0, mm
        ax21.imshow(x0[0], cmap='gray', vmin=0, vmax=170)
        ax22.imshow((mm[0] * (255.0 / np.amax(mm[0]))).astype(np.uint8))
        # mask, frame closer mask
        ax31.imshow(loss.warp_mask[0], cmap='gray', vmin=0, vmax=1)
        ax32.imshow(loss.frame_closer_to_cam_mask[0], cmap='gray', vmin=0, vmax=1)
        # warped img1 -> 0, x1 -> 0
        ax41.imshow(cv2.cvtColor(loss.resampled_img1[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax42.imshow(loss.resampled_depth1[0], cmap='gray', vmin=0, vmax=170)
        # masked warped img1 -> 0, x1 -> 0
        ax51.imshow(cv2.cvtColor(loss.resampled_img1_masked[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax52.imshow(loss.resampled_depth1_masked[0], cmap='gray', vmin=0, vmax=170)
        plt.show()


if __name__ == "__main__":
    test()
