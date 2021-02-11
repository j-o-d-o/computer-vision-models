import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams
from models.dmds.resampler import resampler_with_unstacked_warp
import numpy as np


def construct_rotation_matrix(rot):
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


def warp_it(depth, translation, rotation, intrinsic_mat, intrinsic_mat_inv):
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


class DmdsLoss:
    def __init__(self):
        pass
    
    def norm(self, x):
        return tf.reduce_sum(tf.square(x), axis=-1)
    
    def expand_dims_twice(self, x, dim):
        return tf.expand_dims(tf.expand_dims(x, dim), dim)

    def mm_l12_sparsity(self, obj_motion_map):
        tensor_abs = tf.abs(obj_motion_map)
        mean = tf.stop_gradient(tf.reduce_mean(tensor_abs, axis=[1, 2], keepdims=True))
        return tf.reduce_mean(2 * mean * tf.sqrt(tensor_abs / (mean + 1e-24) + 1))

    def mm_group_smoothness(self, mm):
        grad_dx = mm - tf.roll(mm, 1, 1)
        grad_dy = mm - tf.roll(mm, 1, 2)
        grad_dx = grad_dx[:, 1:, 1:, :]
        grad_dy = grad_dy[:, 1:, 1:, :]
        return tf.reduce_mean(tf.sqrt(1e-24 + tf.square(grad_dx) + tf.square(grad_dy)))

    def depth_edge_aware_smoothness(self, depth_map, img):
        def _gradient_x(img):
            return img[:, :, :-1, :] - img[:, :, 1:, :]
        def _gradient_y(img):
            return img[:, :-1, :, :] - img[:, 1:, :, :]
        # small variance loss to discourage that everything is constant in the depth map
        mean_depth = tf.reduce_mean(depth_map)
        depth_var = tf.reduce_mean(tf.square(depth_map / (mean_depth + 1e-24) - 1.0))
        depth_var = (1.0 / (depth_var + 1e-24))
        # smoothing loss
        disp = 1.0 / depth_map
        disp_dx = _gradient_x(disp)
        disp_dy = _gradient_y(disp)
        img_dx = _gradient_x(img)
        img_dy = _gradient_y(img)
        weights_x = tf.exp(-tf.reduce_mean(tf.abs(img_dx), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(img_dy), 3, keepdims=True))
        smoothness_x = disp_dx * weights_x
        smoothness_y = disp_dy * weights_y
        loss_val_x = tf.reduce_mean(abs(smoothness_x))
        loss_val_y = tf.reduce_mean(abs(smoothness_y))
        alpha_dep = 0.001
        return alpha_dep * (loss_val_x + loss_val_y + depth_var)

    def rgbd_consistency_loss(self, img1, img0_warped, x1, x0_warped, mask):
        # rgb loss
        rgb_l1_diff = tf.abs(img1 - img0_warped)
        rgb_l1_error = tf.reduce_mean(tf.math.multiply_no_nan(rgb_l1_diff, tf.stack([mask]*3, -1)))
        # depth loss (not in paper)
        x1_squeezed = tf.squeeze(x1, axis=3)
        depth_l1_diff = tf.abs(x1_squeezed - x0_warped)
        depth_l1_error = tf.reduce_mean(tf.math.multiply_no_nan(depth_l1_diff, mask))
        # ssim
        ssim = tf.image.ssim(img1, img0_warped, 255.0)
        alpha_rgb = 1.0
        beta_rgb = 3.0
        return alpha_rgb * (rgb_l1_error + depth_l1_error) + beta_rgb * ((1.0 - ssim[0]) * 0.5)

    def motion_field_consistency_loss(self, R, T, mask, px, py):
        # The uneven batch indices are the flipped counter parts to the even batch indices
        mask = mask[::2]
        R_forward = R[::2]
        R_backward = R[1::2]
        R_unit = tf.matmul(R_forward, R_backward)
        identity = tf.eye(3, batch_shape=tf.shape(R_unit)[:-2])
        R_error = tf.reduce_mean(tf.square(R_unit - identity), axis=(1, 2))
        rot1_scale = tf.reduce_mean(tf.square(R_forward - identity), axis=(1, 2))
        rot2_scale = tf.reduce_mean(tf.square(R_backward - identity), axis=(1, 2))
        R_error /= (1e-24 + rot1_scale + rot2_scale)
        R_error = tf.reduce_mean(R_error)

        T_forward = T[::2]
        T_backward = T[1::2]
        T_backward_warped = resampler_with_unstacked_warp(T_backward, tf.stop_gradient(px[::2]), tf.stop_gradient(py[::2]))
        
        rot_field_shape = tf.shape(tf.stack([T_forward]*3, axis=-1)) # stupid way to get the correct shape for the matrix filed for the rotation...
        R_field_backward = tf.broadcast_to(self.expand_dims_twice(R_backward, -3), rot_field_shape)
        r2t1 = tf.matmul(R_field_backward, tf.expand_dims(T_forward, -1))
        r2t1 = tf.squeeze(r2t1, axis=-1)
        trans_zero = r2t1 + T_backward_warped

        # t = tf.math.multiply_no_nan(mask, self.norm(trans_zero))
        t = self.norm(trans_zero)

        translation_error = tf.reduce_mean(t / (1e-24 + self.norm(T_forward) + self.norm(T_backward_warped)))
        
        alpha_cyc = 1.0e-3
        beta_cyc = 5.0e-2

        return alpha_cyc * R_error + beta_cyc * translation_error

    def calc(self, img0, img1, x0, x1, mm, tran, rot, intr):
        # Regulizers for depth (every depthmap is twice in the batches)
        edge_smooth_loss_x0 = self.depth_edge_aware_smoothness(x0, img0) * 0.5
        edge_smooth_loss_x1 = self.depth_edge_aware_smoothness(x1, img1) * 0.5
        # Regulizers for motion map
        sparsity_mm = self.mm_l12_sparsity(mm)
        group_smooth_loss_mm = self.mm_group_smoothness(mm)
        alpha_mot = 1.0
        beta_mot = 0.2
        mm_loss = (alpha_mot * sparsity_mm + group_smooth_loss_mm * beta_mot) * 0.5

        # Construct intrinsics
        K = intr
        K_inv = tf.linalg.inv(K)
        # Construction Rotation Matrix rotation around: [pitch, yaw, roll]
        R = construct_rotation_matrix(rot)
        # Construction Translation image (Object Motion + Ego Motion)
        ego_mm = tf.broadcast_to(self.expand_dims_twice(tran, -2), shape=tf.shape(mm))
        T = mm + ego_mm
        # Warp img0 and x0 into frame 1
        x0_squeezed = tf.squeeze(x0, axis=3)
        px, py, x0_warped, mask = warp_it(x0_squeezed, T, R, K, K_inv)
        img0_warped = resampler_with_unstacked_warp(img0, px, py)
        # Masks out everything that is out of the image, network does not need to learn this masking, cutting gradient
        mask = tf.stop_gradient(mask)
        mask = tf.cast(tf.logical_and(mask, tf.less_equal(x0_warped, x0_squeezed)), tf.float32)

        rgb_loss = self.rgbd_consistency_loss(img1, img0_warped, x1, x0_warped, mask)
        mc_loss = self.motion_field_consistency_loss(R, T, mask, px, py)

        # mask = mask.numpy()
        # mask_3channel = np.stack([mask]*3, axis=-1)
        # x0_squeezed = x0_squeezed.numpy()
        # x0_warped = x0_warped.numpy()
        # x0_warped = mask * x0_warped
        # img0_warped = img0_warped.numpy()
        # img0_warped = mask_3channel * img0_warped
        # f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
        # ax11.imshow(cv2.cvtColor(img0[0].astype(np.uint8), cv2.COLOR_BGR2RGB))
        # ax12.imshow(cv2.cvtColor(img0_warped[0].astype(np.uint8), cv2.COLOR_BGR2RGB))
        # ax21.imshow(x0_squeezed[0], cmap='gray', vmin=0, vmax=170)
        # ax22.imshow(x0_warped[0], cmap='gray', vmin=0, vmax=170)
        # plt.show()

        return mc_loss + rgb_loss + mm_loss + edge_smooth_loss_x0 + edge_smooth_loss_x1


def test():
    from pymongo import MongoClient
    import cv2
    from common.utils import resize_img
    import matplotlib.pyplot as plt

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
        loss.calc(img0, img1, x0, x1, mm, tran, rot, intr)

if __name__ == "__main__":
    test()
