import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block, encoder
from typing import List
from tensorflow.keras.utils import plot_model
from models.semseg import SemsegParams, create_dataset
from models.depth import create_model as create_depth_model
from common.utils.tflite_convert import tflite_convert
from common.utils.set_weights import set_weights


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
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg_out", kernel_regularizer=l2(l=0.0001))(x)
    
    model = Model(inputs=[inp], outputs=semseg_map)
    if weights_path is not None:
        set_weights(weights_path, model)

    return model


if __name__ == "__main__":
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
