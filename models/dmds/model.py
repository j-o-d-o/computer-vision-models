import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from tensorflow.python.keras.engine import data_adapter
from models.dmds import DmdsParams
from models.dmds.convert import create_dataset
from common.utils import tflite_convert, resize_img
from common.layers import encoder, upsample_block, bottle_neck_block
from models.depth.model import create_model as create_depth_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pygame
from tensorflow.python.eager import backprop


class DmdsModel(Model):
    def init_file_writer(self, log_dir):
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.train_step_counter = 0

        self.display = pygame.display.set_mode((640*2, 256*3), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    def _cmap(self, depth_map, max_val = 140.0, to_255 = True, swap_axes = True):
        npumpy_depth_map = depth_map.numpy()
        if max_val == "max":
            npumpy_depth_map /= np.amax(npumpy_depth_map)
        else:
            npumpy_depth_map /= max_val
        npumpy_depth_map = np.clip(npumpy_depth_map, 0.0, 1.0)
        if to_255:
            npumpy_depth_map *= 255.0
        if swap_axes:
            npumpy_depth_map = npumpy_depth_map.swapaxes(0, 1)
        depth_stack = [npumpy_depth_map.astype(np.uint8)]
        npumpy_depth_map = np.concatenate(depth_stack * 3, axis=-1)
        return npumpy_depth_map

    def train_step(self, data):
        def rgb_img(img):
            np_img = img.numpy()
            c0 = tf.unstack(np_img, axis=-1)
            return tf.stack([c0[2], c0[1], c0[0]], axis=-1)

        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            combined_input = (
                tf.concat([input_data[0], input_data[1]], axis=0),
                tf.concat([input_data[1], input_data[0]], axis=0),
                tf.concat([input_data[2], input_data[2]], axis=0)
            )
            y_pred = self(combined_input, training=True)
            depth_stack, flipped_depth_stack, mm_stack, tran_stack, rot_stack, _ = y_pred

            loss_val = self.custom_loss.calc(
                combined_input[0],
                combined_input[1],
                depth_stack,
                flipped_depth_stack,
                mm_stack,
                tran_stack,
                rot_stack,
                combined_input[2],
                gt[0],
                gt[1],
                self.train_step_counter)

        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        # Display images
        surface_img0 = pygame.surfarray.make_surface(combined_input[0][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img0, (0, 0))
        surface_img1 = pygame.surfarray.make_surface(combined_input[1][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img1, (640, 0))

        surface_x0 = pygame.surfarray.make_surface(self._cmap(depth_stack[0], max_val="max"))
        self.display.blit(surface_x0, (0, 256))
        surface_weights = pygame.surfarray.make_surface(self._cmap(tf.expand_dims(self.custom_loss.depth_weights[0], axis=-1), max_val=1.0))
        self.display.blit(surface_weights, (640, 256))

        surface_warped = pygame.surfarray.make_surface(self.custom_loss.resampled_img1[0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_warped, (0, 256*2))
        surface_mask = pygame.surfarray.make_surface(self._cmap(self.custom_loss.warp_mask[0], max_val=1.0))
        self.display.blit(surface_mask, (640, 256*2))

        if self.train_step_counter % 100 == 0:
            pygame.image.save(self.display, f"{self.log_dir}/train_result_{self.train_step_counter}.png")

        pygame.display.flip()

        # Using the file writer, log images
        tf.summary.experimental.set_step(int(self.train_step_counter / 40))
        with self.file_writer.as_default():
            # tf.summary.image("img0", rgb_img(combined_input[0]), max_outputs=10)

            tf.summary.histogram("depth_hist", depth_stack)
            tf.summary.histogram("tran", tran_stack)
            tf.summary.histogram("rot", rot_stack)

        self.train_step_counter += 1

        self.custom_loss.loss_vals["sum"] = loss_val
        return self.custom_loss.loss_vals
    
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        combined_input = (
            tf.concat([input_data[0], input_data[1]], axis=0),
            tf.concat([input_data[1], input_data[0]], axis=0),
            tf.concat([input_data[2], input_data[2]], axis=0)
        )
        y_pred = self(combined_input, training=True)

        depth_stack, flipped_depth_stack, mm_stack, tran_stack, rot_stack, _ = y_pred

        loss_val = self.custom_loss.calc(combined_input[0], combined_input[1], depth_stack, flipped_depth_stack, mm_stack, tran_stack, rot_stack, combined_input[2], gt[0], gt[1], self.train_step_counter)


        self.custom_loss.loss_vals["sum"] = loss_val
        return self.custom_loss.loss_vals


class ScaleConstraint(tf.keras.constraints.Constraint):
    """The weight tensors will be constrained to not fall below constraint_minimum, this is used for the scale variables in DMLearner/motion_field_net."""

    def __init__(self, constraint_minimum=0.01):
        self.constraint_minimum = constraint_minimum

    def __call__(self, w):
        return tf.nn.relu(w - self.constraint_minimum) + self.constraint_minimum

    def get_config(self):
        return {'constraint_minimum': self.constraint_minimum}


def create_model(input_height: int, input_width: int, depth_model_path: str = None) -> tf.keras.Model:
    intr = Input(shape=(3, 3))
    input_t0 = Input(shape=(input_height, input_width, 3))
    rescaled_input_t0 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(input_t0)

    input_t1 = Input(shape=(input_height, input_width, 3))
    rescaled_input_t1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(input_t1)

    # Depth Maps
    # =======================================
    depth_model = create_depth_model(input_height, input_width, depth_model_path)

    x0 = depth_model(input_t0)
    x1 = depth_model(input_t1)

    # Motion Network
    # =======================================
    inp_t0 = Concatenate()([rescaled_input_t0, x0])
    inp_t1 = Concatenate()([rescaled_input_t1, x1])
    x = Concatenate()([inp_t0, inp_t1])
    x, fms = encoder(6, x, filter_scaling=1.5, namescope="mm/", output_scaled_down=True)

    ego_motion = Conv2D(64, kernel_size=3, strides=2, use_bias=False)(fms[-1])
    ego_motion = LayerNormalization()(ego_motion)
    ego_motion = ReLU(6.)(ego_motion)
    ego_motion = Conv2D(64, kernel_size=3, strides=2, use_bias=False)(ego_motion)
    ego_motion = LayerNormalization()(ego_motion)
    ego_motion = ReLU(6.)(ego_motion)
    ego_motion = Flatten()(ego_motion)
    ego_motion = Dropout(0.3)(ego_motion)

    rot = Dense(32, use_bias=False)(ego_motion)
    rot = LayerNormalization()(rot)
    rot = ReLU(6.)(rot)
    rot = Dropout(0.2)(rot)
    rot = Dense(16, use_bias=False)(rot)
    rot = LayerNormalization()(rot)
    rot = ReLU(6.)(rot)
    rot = Dropout(0.1)(rot)
    rot = Dense(3, activation=None)(rot)

    tran = Dense(32, use_bias=False)(ego_motion)
    tran = LayerNormalization()(tran)
    tran = ReLU(6.)(tran)
    tran = Dropout(0.2)(tran)
    tran = Dense(16, use_bias=False)(tran)
    tran = LayerNormalization()(tran)
    tran = ReLU(6.)(tran)
    tran = Dropout(0.1)(tran)
    tran = Dense(3)(tran)

    mm = Conv2D(8, (3, 3), padding="same", name="motion_map_conv2d")(x)
    mm = LayerNormalization()(mm)
    mm = ReLU()(mm)
    mm = Conv2D(3, kernel_size=1, padding="same", name="motion_map")(mm)

    mm_scaling = Conv2D(1, (1, 1), use_bias=False, kernel_constraint=ScaleConstraint(0.01), name='motion_field_net/mm_scaling')
    mm = Concatenate(axis=-1, name='motion_field_net/scaled_mm')([
        mm_scaling(tf.expand_dims(mm[:, :, :, 0], axis=-1)),
        mm_scaling(tf.expand_dims(mm[:, :, :, 1], axis=-1)),
        mm_scaling(tf.expand_dims(mm[:, :, :, 2], axis=-1))])

    rot_scaling = Dense(1, use_bias=False, kernel_constraint=ScaleConstraint(), kernel_initializer=initializers.constant(0.01), name='motion_field_net/scaling_rot')
    rot = Concatenate(axis=-1, name='motion_field_net/scaled_rotation')([
        rot_scaling(tf.expand_dims(rot[:, 0], axis=-1)),
        rot_scaling(tf.expand_dims(rot[:, 1], axis=-1)),
        rot_scaling(tf.expand_dims(rot[:, 2], axis=-1))])
    
    tran_scaling = Dense(1, use_bias=False, kernel_constraint=ScaleConstraint(), kernel_initializer=initializers.constant(0.01), name='motion_field_net/scaling_tran')
    tran = Concatenate(axis=-1, name='motion_field_net/scaled_translation')([
        tran_scaling(tf.expand_dims(tran[:, 0], axis=-1)),
        tran_scaling(tf.expand_dims(tran[:, 1], axis=-1)),
        tran_scaling(tf.expand_dims(tran[:, 2], axis=-1))])

    return DmdsModel(inputs=[input_t0, input_t1, intr], outputs=[x0, x1, mm, tran, rot, intr])


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, params.LOAD_DEPTH_MODEL)
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))
