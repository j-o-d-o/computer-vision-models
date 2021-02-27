import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block, encoder
from numba.typed import List
from tensorflow.keras.utils import plot_model
from models.semseg import SemsegParams, create_dataset
from models.depth import create_model as create_depth_model
from common.utils.tflite_convert import tflite_convert
from common.utils.set_weights import set_weights
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from common.utils import to_3channel


class SemsegModel(Model):
    def init_save_dir(self, save_dir):
        self.save_dir = save_dir
        self.file_writer = tf.summary.create_file_writer(save_dir)
        self.train_step_counter = 0

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    @property
    def metrics(self):
        return list(self.custom_loss.metrics.values())

    def train_step(self, data):
        self.train_step_counter += 1
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_mask = gt[0]
        pos_mask = gt[1]

        with backprop.GradientTape() as tape:
            semseg_pred = self(input_data, training=True)
            loss_val = self.custom_loss.calc(input_data, semseg_mask, pos_mask, semseg_pred)
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        # Using the file writer, log images
        if self.train_step_counter % 200 == 0:
            tf.summary.experimental.set_step(self.train_step_counter)
            with self.file_writer.as_default():
                inp_img = cv2.cvtColor(input_data[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
                semseg_true_img = cv2.cvtColor(to_3channel(semseg_mask[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB)
                semseg_pred_img = cv2.cvtColor(to_3channel(semseg_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB)

                tf.summary.image("inp", np.expand_dims(inp_img, axis=0), max_outputs=80)
                tf.summary.image("true", np.expand_dims(semseg_true_img, axis=0), max_outputs=80)
                tf.summary.image("pred", np.expand_dims(semseg_pred_img, axis=0), max_outputs=80)

        return_val = {}
        for key, item in self.custom_loss.metrics.items():
            return_val[key] = item.result()
        return return_val

    def test_step(self, data):
        # self.train_step_counter = 0
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_mask = gt[0]
        pos_mask = gt[1]

        semseg_pred = self(input_data, training=False)
        loss_val = self.custom_loss.calc(input_data, semseg_mask, pos_mask, semseg_pred)

        return_val = {}
        for key, item in self.custom_loss.metrics.items():
            return_val[f"{key}_val"] = item.result()
        return return_val


def create_model(input_height: int, input_width: int, weights_path: str = None) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    inp = Input(shape=(input_height, input_width, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)

    x, _ = encoder(8, inp_rescaled)
    x = Conv2D(8, (3, 3), padding="same", name="semseg_head_conv2d", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg_out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(x)

    model = SemsegModel(inputs=[inp], outputs=semseg_map)
    if weights_path is not None:
        set_weights(weights_path, model, {"SemsegModel": SemsegModel})

    return model


if __name__ == "__main__":
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
