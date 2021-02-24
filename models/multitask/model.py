import tensorflow as tf
import pygame
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block, encoder
from typing import List
from tensorflow.keras.utils import plot_model
from models.multitask import MultitaskParams, create_dataset
from common.utils.tflite_convert import tflite_convert
from common.utils.set_weights import set_weights
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from common.utils import to_3channel, cmap_depth


class MultitaskModel(Model):
    def init_save_path(self, save_path):
        self.save_path = save_path
        self.train_step_counter = 0
        self.display = pygame.display.set_mode((640*2, 256*3), pygame.HWSURFACE | pygame.DOUBLEBUF)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    @property
    def metrics(self):
        return list(self.custom_loss.metrics.values())

    def show_data(self, inp, y_true_semseg, y_pred_semseg, semseg_valid, y_true_depth, y_pred_depth, depth_valid):
        filler = np.zeros((inp[0].shape))

        inp_img = cv2.cvtColor(inp[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_img = pygame.surfarray.make_surface(inp_img)
        self.display.blit(surface_img, (0, 0))
        # semseg
        if semseg_valid:
            semseg_true = cv2.cvtColor(to_3channel(y_true_semseg[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            surface_y_true = pygame.surfarray.make_surface(semseg_true)
        else:
            surface_y_true = pygame.surfarray.make_surface(filler)
        self.display.blit(surface_y_true, (0, 256))
        semseg_pred = cv2.cvtColor(to_3channel(y_pred_semseg[0].numpy(), list(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        surface_y_pred = pygame.surfarray.make_surface(semseg_pred)
        self.display.blit(surface_y_pred, (0, 256*2))
        # depth
        if depth_valid:
            surface_y_true = pygame.surfarray.make_surface(cmap_depth(y_true_depth[0], vmin=0.1, vmax=255.0).swapaxes(0, 1))
        else:
            surface_y_true = pygame.surfarray.make_surface(filler)
        self.display.blit(surface_y_true, (640, 256*1))
        surface_y_pred = pygame.surfarray.make_surface(cmap_depth(y_pred_depth[0], vmin=0.1, vmax=255.0).swapaxes(0, 1))
        self.display.blit(surface_y_pred, (640, 256*2))

        if self.train_step_counter % 500 == 0:
            pygame.image.save(self.display, f"{self.save_path}/train_result_{self.train_step_counter}.png")

        pygame.display.flip()

    def train_step(self, data):
        self.train_step_counter += 1

        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_true = gt[0]
        semseg_valid = tf.reduce_all(gt[1]) == True
        depth_true = gt[2]
        depth_valid = tf.reduce_all(gt[3]) == True
        pos_mask = gt[4]

        base_model: Model = self.get_layer("base/model")
        base_model.trainable = True
        refine_model: Model = self.get_layer("refine/model")
        refine_model.trainable = True

        semseg_head: Model = self.get_layer("semseg_head/model")
        semseg_head.trainable = semseg_valid
        depth_head: Model = self.get_layer("depth_head/model")
        depth_head.trainable = depth_valid


        total_loss = 0
        trainable_vars = base_model.trainable_variables
        with backprop.GradientTape() as tape:
            semseg_pred, depth_pred = self(input_data)
            if semseg_valid:
                semseg_loss = self.custom_loss.calc_semseg(semseg_true, pos_mask, semseg_pred)
                total_loss += semseg_loss
                trainable_vars = [*trainable_vars, *semseg_head.trainable_variables]
            if depth_valid:
                depth_loss = self.custom_loss.calc_depth(depth_true, tf.squeeze(depth_pred, axis=-1))
                total_loss += depth_loss
                trainable_vars = [*trainable_vars, *depth_head.trainable_variables]

        grads = tape.gradient(total_loss, trainable_vars)
        # capped_grads = [MyCapper(g) for g in grads]
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        self.show_data(input_data, semseg_true, semseg_pred, semseg_valid, depth_true, tf.squeeze(depth_pred, axis=-1), depth_valid)

        return_val = {}
        for key, item in self.custom_loss.metrics.items():
            return_val[key] = item.result()
        return return_val

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_true = gt[0]
        semseg_valid = gt[1]
        depth_true = gt[2]
        depth_valid = gt[3]
        pos_mask = gt[4]

        with backprop.GradientTape() as tape:
            semseg_pred, depth_pred = self(input_data, training=False)
            if semseg_valid:
                semseg_loss = self.custom_loss.calc_semseg(semseg_true, pos_mask, semseg_pred)
            if depth_valid:
                depth_loss = self.custom_loss.calc_depth(depth_true, tf.squeeze(depth_pred, axis=-1))

        return_val = {}
        for key, item in self.custom_loss.metrics.items():
            return_val[f"{key}_val"] = item.result()
        return return_val

def _create_refine_head(input_height: int, input_width: int, channels: int) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, channels))
    refi = Conv2D(12, (3, 3), use_bias=False, padding="same", name="refined/conv2d_0")(inp)
    refi = BatchNormalization(name="refine/batch_norm_0")(refi)
    refi = ReLU()(refi)
    refi = Conv2D(8, (3, 3), use_bias=False, padding="same", name="refine/conv2d_1")(refi)
    refi = BatchNormalization(name="refine/batch_norm_1")(refi)
    refi = ReLU()(refi)
    refined_semseg_img = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="refine/out_semseg", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(refi)
    refined_depth_img = Conv2D(1, kernel_size=1, padding="same", activation="relu", use_bias=True, name="refine/out_depth")(refi)
    return Model(inputs=[inp], outputs=[refined_semseg_img, refined_depth_img], name="refine/model")

def _create_depth_head(input_height: int, input_width: int, channels: int, path = None) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, channels))
    di = Conv2D(12, (3, 3), use_bias=False, padding="same", name="depth_head/conv2d_0")(inp)
    di = BatchNormalization(name="depth_head/batch_norm_0")(di)
    di = ReLU()(di)
    di = Conv2D(8, (3, 3), use_bias=False, padding="same", name="depth_head/conv2d_1")(di)
    di = BatchNormalization(name="depth_head/batch_norm_1")(di)
    di = ReLU()(di)
    di = Conv2D(1, kernel_size=1, padding="same", activation="relu", use_bias=True, name="depth_head/out")(di)
    DepthModel = Model(inputs=[inp], outputs=di, name="depth_head/model")
    if path is not None:
        set_weights(path, DepthModel, {"MultitaskModel", MultitaskModel})
    return DepthModel

def _create_semseg_head(input_height: int, input_width: int, channels: int, path = None) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, channels))
    si = Conv2D(12, (3, 3), padding="same", name="semseg_head/conv2d_0", use_bias=False)(inp)
    si = BatchNormalization(name="semseg_head/batch_norm_0")(si)
    si = ReLU()(si)
    si = Conv2D(8, (3, 3), padding="same", name="semseg_head/conv2d_1", use_bias=False)(si)
    si = BatchNormalization(name="semseg_head/batch_norm_1")(si)
    si = ReLU()(si)
    si = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg_head/out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(si)
    SemsegModel = Model(inputs=[inp], outputs=si, name="semseg_head/model")
    if path is not None:
        set_weights(path, SemsegModel, {"MultitaskModel", MultitaskModel})
    return SemsegModel

def _create_base_model(input_height: int, input_width: int, path = None) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, 3))
    x, _ = encoder(12, inp)
    BaseModel = Model(inputs=[inp], outputs=x, name="base/model")
    if path is not None:
        set_weights(path, BaseModel, {"MultitaskModel", MultitaskModel})
    return BaseModel

def create_model(input_height: int, input_width: int, path_base=None, path_semseg=None, path_depth=None) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)
    
    # Base Model
    # ------------------------------
    BaseModel = _create_base_model(input_height, input_width, path_base)
    x = BaseModel([inp_rescaled])

    # Semseg Head
    # ------------------------------
    SemsegHead = _create_semseg_head(input_height, input_width, x.shape[-1], path_semseg)
    semseg_img = SemsegHead([x])

    # Depth Head
    # ------------------------------
    DepthHead = _create_depth_head(input_height, input_width, x.shape[-1], path_semseg)
    depth_img = DepthHead([x])

    # Refine Depth & Semseg
    # ------------------------------
    concat = Concatenate()([semseg_img, depth_img])
    RefineHead = _create_refine_head(input_height, input_width, concat.shape[-1]) 
    refined_semseg_img, refined_depth_img = RefineHead(concat)

    return MultitaskModel(inputs=[inp], outputs=[refined_semseg_img, refined_depth_img], name="multitask_model")


if __name__ == "__main__":
    params = MultitaskParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/multitask_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
