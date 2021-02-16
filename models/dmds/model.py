import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda
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


# Creating custom train_step() in order to call loss function with all needed parameters
class DmdsModel(Model):
    def init_file_writer(self, logdir):
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.train_step_counter = 0

        self.display = pygame.display.set_mode((320*2, 128*3), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    def train_step(self, data):
        def rgb_img(img):
            img0 = (img) / 255.0
            c0 = tf.unstack(img0, axis=-1)
            return tf.stack([c0[2], c0[1], c0[0]], axis=-1)

        def scale_to_max(img):
            return img / tf.reduce_max(img)

        data = data_adapter.expand_1d(data)
        # [0: x0, 1: x1, 2: mm, 3: rot, 4: tran, 5: intr]
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(input_data, training=True)
            y_pred_inv = self((input_data[1], input_data[0], input_data[2]), training=True)

            img0 = input_data[0]
            img1 = input_data[1]
            x0, x1, mm, tran, rot, intr = y_pred
            _, _, mm_inv, tran_inv, rot_inv, _ = y_pred_inv

            loss_val = self.custom_loss.calc(img0, img1, x0, x1, mm, mm_inv, tran, tran_inv, rot, rot_inv, intr, gt[0], gt[1])

        grads = tape.gradient(loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Display images
        surface_img0 = pygame.surfarray.make_surface(input_data[0][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img0, (0, 0))
        surface_img1 = pygame.surfarray.make_surface(input_data[1][0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_img1, (320, 0))

        surface_x0 = pygame.surfarray.make_surface(np.concatenate([(scale_to_max(x0[0]).numpy() * 255.0).astype(np.uint8).swapaxes(0, 1)] * 3, axis=-1))
        self.display.blit(surface_x0, (0, 128))
        surface_x1 = pygame.surfarray.make_surface(np.concatenate([(scale_to_max(x0[0]).numpy() * 255.0).astype(np.uint8).swapaxes(0, 1)] * 3, axis=-1))
        self.display.blit(surface_x1, (320, 128))

        surface_warped = pygame.surfarray.make_surface(self.custom_loss.resampled_img1[0].numpy().astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_warped, (0, 128*2))
        surface_warped = pygame.surfarray.make_surface((scale_to_max(mm[0]).numpy() * 255.0).astype(np.uint8).swapaxes(0, 1))
        self.display.blit(surface_warped, (320, 128*2))

        pygame.display.flip()

        # Using the file writer, log the reshaped image.
        tf.summary.experimental.set_step(int(self.train_step_counter / 40))
        with self.file_writer.as_default():
            tf.summary.image("img0", rgb_img(input_data[0]), max_outputs=5)
            tf.summary.image("img1", rgb_img(input_data[1]), max_outputs=5)
            tf.summary.image("resampled_img1", rgb_img(self.custom_loss.resampled_img1), max_outputs=5)

            tf.summary.image("depth0", scale_to_max(x0), max_outputs=5)
            tf.summary.image("resampled_depth1", scale_to_max(self.custom_loss.resampled_depth1), max_outputs=5)
            tf.summary.image("mm", scale_to_max(mm), max_outputs=5)

            tf.summary.image("warp_mask", self.custom_loss.warp_mask, max_outputs=5)
            tf.summary.image("frame_closer_mask", self.custom_loss.frame_closer_to_cam_mask, max_outputs=5)

            tf.summary.histogram("depth_hist", x0)
            tf.summary.histogram("tran", tran)
            tf.summary.histogram("tran_inv", tran_inv)
            tf.summary.histogram("rot", rot)
            tf.summary.histogram("rot_inv", rot_inv)

        self.train_step_counter += 1

        self.custom_loss.loss_vals["sum"] = loss_val
        return self.custom_loss.loss_vals
    
    def test_step(self, data):
        self.train_step_counter = 0

        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(input_data, training=True)
        y_pred_inv = self((input_data[1], input_data[0], input_data[2]), training=True)

        img0 = input_data[0]
        img1 = input_data[1]
        x0, x1, mm, tran, rot, intr = y_pred
        _, _, mm_inv, tran_inv, rot_inv, _ = y_pred_inv

        loss_val = self.custom_loss.calc(img0, img1, x0, x1, mm, mm_inv, tran, tran_inv, rot, rot_inv, intr, gt[0], gt[1])

        self.custom_loss.loss_vals["sum"] = loss_val
        return self.custom_loss.loss_vals


class ScaleConstraint(tf.keras.constraints.Constraint):
    """The weight tensors will be constrained to not fall below constraint_minimum, this is used for the scale variables in DMLearner/motion_field_net."""

    def __init__(self, constraint_minimum=0.001):
        self.constraint_minimum = constraint_minimum

    def __call__(self, w):
        return tf.nn.relu(w - self.constraint_minimum) + self.constraint_minimum

    def get_config(self):
        return {'constraint_minimum': self.constraint_minimum}


def create_model(input_height: int, input_width: int, depth_model_path: str = None) -> tf.keras.Model:
    intr = Input(shape=(3, 3))
    input_t0 = Input(shape=(input_height, input_width, 3))
    input_t1 = Input(shape=(input_height, input_width, 3))

    # Depth Maps
    # =======================================
    depth_model = create_depth_model(input_height, input_width, depth_model_path)

    x0 = depth_model(input_t0)
    x1 = depth_model(input_t1)

    # Motion Network
    # =======================================
    inp_t0 = Concatenate()([input_t0, x0])
    inp_t1 = Concatenate()([input_t1, x1])
    x = Concatenate()([inp_t0, inp_t1])
    x, fms = encoder(8, x, namescope="mm")

    bottleneck = Conv2D(128, kernel_size=3, strides=2, name="down_you_go_0")(fms[-1])
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = ReLU()(bottleneck)
    bottleneck = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(bottleneck)
    background_motion = Conv2D(6, [1, 1], strides=1, name='motion_field_net/background_motion')(bottleneck)

    rot = Lambda(lambda y: y[:, 0, 0, :3])(background_motion)
    tran = Lambda(lambda y: y[:, 0, 0, 3:])(background_motion)

    mm = Conv2D(16, (3, 3), padding="same", name="mm_conv2d", use_bias=False)(x)
    mm = BatchNormalization(name="mm_batchnorm")(mm)
    mm = ReLU()(mm)
    mm = Conv2D(3, kernel_size=1, padding="same", name="motion_map")(mm)

    trans_scale = Conv2D(1, (1, 1), use_bias=False, kernel_constraint=ScaleConstraint(), kernel_initializer=initializers.constant(0.01), name='motion_field_net/translation_scale')
    mm = Concatenate(axis=-1, name='motion_field_net/scaled_residual_translation')([trans_scale(tf.expand_dims(mm[:, :, :, 0], axis=-1)),
                                                                                    trans_scale(tf.expand_dims(mm[:, :, :, 1], axis=-1)),
                                                                                    trans_scale(tf.expand_dims(mm[:, :, :, 2], axis=-1))])

    rot_scale = Dense(1, use_bias=False, kernel_constraint=ScaleConstraint(), kernel_initializer=initializers.constant(0.01), name='motion_field_net/rotation_scale')
    rot = Concatenate(axis=-1, name='motion_field_net/scaled_rotation')([rot_scale(tf.expand_dims(rot[:, 0], axis=-1)),
                                                                         rot_scale(tf.expand_dims(rot[:, 1], axis=-1)),
                                                                         rot_scale(tf.expand_dims(rot[:, 2], axis=-1))])
    tran = Concatenate(axis=-1, name='motion_field_net/scaled_rotation2')([rot_scale(tf.expand_dims(tran[:, 0], axis=-1)),
                                                                          rot_scale(tf.expand_dims(tran[:, 1], axis=-1)),
                                                                          rot_scale(tf.expand_dims(tran[:, 2], axis=-1))])

    return DmdsModel(inputs=[input_t0, input_t1, intr], outputs=[x0, x1, mm, tran, rot, intr])


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, "/home/computer-vision-models/trained_models/depth_ds_2021-02-15-121653/tf_model_0")
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))
