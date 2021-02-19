import tensorflow as tf
from tensorflow.keras.losses import Loss
from models.dmds.params import DmdsParams
import numpy as np
from pymongo import MongoClient
import cv2
import matplotlib.pyplot as plt
from common.utils import resize_img
# from models.dmds.resampler import resampler_with_unstacked_warp
from models.dmds_ref import regularizers, resampler, transform_depth_map, consistency_losses, intrinsics_utils
from data.driving_stereo_depth import fill_depth_data
from models.depth.loss import DepthLoss


class DmdsLoss:
    def __init__(self, params):
        self.params: DmdsParams = params

        self.loss_vals = {
            # "depth_abs": 0,
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
        self.warp_mask = None
        self.depth_weights = None

    def calc(self, rgb, flipped_rgb, predicted_depth, flipped_predicted_depth, residual_translation, background_translation, rotation, intrinsics_mat, gt_x0, gt_x1, step_number):
        # residual_translation *= tf.math.minimum(1.0, step_number / 1000.0)
        # bg_translation_field = tf.broadcast_to(consistency_losses._expand_dims_twice(translation_stack, -2), shape=tf.shape(mm_stack))
        # bg_flipped_translation_field = tf.broadcast_to(consistency_losses._expand_dims_twice(flipped_translation_stack, -2), shape=tf.shape(mm_stack))
        # translation_field = mm_stack + bg_translation_field

        # expand dims to be able to add it to the resudial_translation which has image dims and size
        background_translation = consistency_losses._expand_dims_twice(background_translation, -2)

        # resize since we output a smaller resudial_translation map
        rgb_shape = rgb.shape[1:3]
        residual_translation = tf.image.resize(residual_translation, rgb_shape,  method='nearest')

        # Data generation
        # -------------------------------
        residual_translation_split = tf.split(residual_translation, 2, axis=0)
        flipped_residual_translation = tf.concat(residual_translation_split[::-1], axis=0)

        background_translation_split = tf.split(background_translation, 2, axis=0)
        flipped_background_translation = tf.concat(background_translation_split[::-1], axis=0)
    
        translation = residual_translation + background_translation
        flipped_translation = (flipped_residual_translation + flipped_background_translation)

        rotation_split = tf.split(rotation, 2, axis=0)
        flipped_rotation = tf.concat(rotation_split[::-1], axis=0)

        intrinsics_mat_inv = intrinsics_utils.invert_intrinsics_matrix(intrinsics_mat)

        # Depth regulizers
        # -------------------------------
        mean_depth = tf.reduce_mean(predicted_depth)
        depth_var = tf.reduce_mean(tf.square(predicted_depth / mean_depth - 1.0))
        self.loss_vals["depth_var"] = (1.0 / depth_var) * self.params.var_depth

        disp = 1.0 / predicted_depth
        mean_disp = tf.reduce_mean(disp, axis=[1, 2, 3], keepdims=True)
        self.loss_vals["depth_smooth"] = regularizers.joint_bilateral_smoothing(disp / mean_disp, rgb) * self.params.depth_smoothing

        # Motionmap regulizers
        # -------------------------------
        normalized_trans = regularizers.normalize_motion_map(residual_translation, translation)
        self.loss_vals["mm_smooth"] = regularizers.l1smoothness(normalized_trans) * self.params.mot_smoothing
        self.loss_vals["mm_sparsity"] = regularizers.sqrt_sparsity(normalized_trans) * self.params.mot_drift

        # Cyclic and RGB Loss
        # -------------------------------
        transformed_depth = transform_depth_map.using_motion_vector(
            tf.squeeze(predicted_depth, axis=-1),
            translation,
            rotation,
            intrinsics_mat,
            intrinsics_mat_inv
        )

        flipped_predicted_depth = tf.squeeze(flipped_predicted_depth, axis=-1)
        flipped_predicted_depth = tf.stop_gradient(flipped_predicted_depth) # TODO: Check if we do not do this

        loss_endpoints = (consistency_losses.rgbd_and_motion_consistency_loss(
            transformed_depth,
            rgb,
            flipped_predicted_depth,
            flipped_rgb,
            rotation,
            translation,
            flipped_rotation,
            flipped_translation,
            # validity_mask=validity_mask
        ))

        self.loss_vals["depth"] = loss_endpoints['depth_error'] * self.params.depth_cons
        self.loss_vals["rgb"] = loss_endpoints['rgb_error'] * self.params.rgb_cons
        self.loss_vals["ssim"] = 0.5 * loss_endpoints['ssim_error'] * self.params.ssim_cons
        self.loss_vals["rot"] = loss_endpoints['rotation_error'] * self.params.rot_cyc
        self.loss_vals["tran"] = loss_endpoints['translation_error'] * self.params.tran_cyc

        self.resampled_img1 = loss_endpoints['warped_images']
        self.warp_mask = loss_endpoints['frame1_closer_to_camera']
        self.depth_weights = loss_endpoints['depth_proximity_weight']

        result = 0
        for key in self.loss_vals.keys():
            if key != "sum":
                result += self.loss_vals[key]

        return result


def test():
    params = DmdsParams()
    loss = DmdsLoss(params)
    batch_size = 2

    client = MongoClient("mongodb://localhost:27017")
    collection = client["depth"]["driving_stereo"]
    documents = collection.find({}).limit(10).skip(300)
    documents = list(documents)
    for i in range(0, len(documents)-1):
        intr = np.array([
            [375.0,  0.0, 160.0],
            [ 0.0, 375.0, 128.0],
            [ 0.0,   0.0,   1.0]
        ], dtype=np.float32)
        intr = np.stack([intr]*batch_size)
        intr_stack = np.concatenate([intr, intr], axis=0)

        img0 = cv2.imdecode(np.frombuffer(documents[i]["img"], np.uint8), cv2.IMREAD_COLOR)
        img0, _ = resize_img(img0, 320, 128, 0)
        img0 = np.stack([img0]*batch_size, axis=0)
        img0 = img0.astype(np.float32)
        img0 /= 255.0

        img1 = cv2.imdecode(np.frombuffer(documents[i+1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img1, _ = resize_img(img1, 320, 128, 0)
        img1 = np.stack([img1]*batch_size, axis=0)
        img1 = img1.astype(np.float32)
        img1 /= 255.0

        rgb_stack = np.concatenate([img0, img1], axis=0)
        rgb_stack_flipped = np.concatenate([img1, img0], axis=0)

        # create gt depth_maps
        x0 = cv2.imdecode(np.frombuffer(documents[i]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x0, _ = resize_img(x0, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x0 = fill_depth_data(x0)
        x0 = np.expand_dims(x0, axis=-1)
        x0 = np.stack([x0]*batch_size, axis=0)
        x0 = x0.astype(np.float32)
        x0 /= 255.0

        x1 = cv2.imdecode(np.frombuffer(documents[i+1]["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
        x1, _ = resize_img(x1, 320, 128, 0, interpolation=cv2.INTER_NEAREST)
        x1 = fill_depth_data(x1)
        x1 = np.expand_dims(x1, axis=-1)
        x1 = np.stack([x1]*batch_size, axis=0)
        x1 = x1.astype(np.float32)
        x1 /= 255.0

        depth_stack = np.concatenate([x0, x1], axis=0)
        depth_stack_flipped = np.concatenate([x1, x0], axis=0)

        mm = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm[:, 23:85, 132:320, :] = [0.0, 0.0, 0.0]

        mm_inv = np.zeros((*img0.shape[:-1], 3), dtype=np.float32)
        mm_inv[:, 23:85, 172:320, :] = [0.0, 0.0, 0.0]
        mm_stack = np.concatenate([mm, mm_inv], axis=0)

        rot = np.zeros((batch_size, 3), dtype=np.float32)
        rot[:,] = np.array([-0.0, 0.0, 0.1])
        rot_inv = np.zeros((batch_size, 3), dtype=np.float32)
        rot_inv[:,] = np.array([0.0, 0.0, -0.1])

        rot_stack = np.concatenate([rot, rot_inv], axis=0)

        tran = np.zeros((batch_size, 3), dtype=np.float32)
        tran[:,] = np.array([3.0, 0.0, 0]) # [left,right | up,down | forward,backward]
        tran_inv = np.zeros((batch_size, 3), dtype=np.float32)
        tran_inv[:,] = np.array([-3.0, 0.0, 0])

        tran_stack = np.concatenate([tran, tran_inv], axis=0)

        loss.calc(rgb_stack, rgb_stack_flipped, depth_stack, depth_stack_flipped, mm_stack, tran_stack, rot_stack, intr_stack, x0, x1, 0)

        for key in loss.loss_vals.keys():
            print(f"{key}: {loss.loss_vals[key].numpy()}")
        print(" - - - - - - - -")

        for i in range(len(rgb_stack)):
            f, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2)
            # img0, img1
            ax11.imshow(cv2.cvtColor((rgb_stack[i] * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax12.imshow(cv2.cvtColor((rgb_stack_flipped[i] * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB))
            # x0, mm
            ax21.imshow(depth_stack[i], cmap='gray', vmin=0, vmax=170)
            ax22.imshow((mm_stack[i] * (255.0 / np.amax(mm_stack[i]))).astype(np.uint8))
            # mask, frame closer mask
            ax31.imshow(loss.warp_mask[i], cmap='gray', vmin=0, vmax=1)
            ax32.imshow(cv2.cvtColor((loss.resampled_img1[i].numpy() * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()


if __name__ == "__main__":
    test()
