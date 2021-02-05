import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from data.semseg_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block
from typing import List
from tensorflow.keras.utils import plot_model
from models.semseg import SemsegParams, create_dataset
from common.utils import tflite_convert


def encoder(filters: int, input_tensor: tf.Tensor, base_model_weight: tf.keras.Model = None):
    fms = []
    x = input_tensor
    x = bottle_neck_block(f"prior_downsample", x, filters, downsample = True)
    fms.append(x)

    for i in range(4):
        x = bottle_neck_block(f"down_{i*2  }", x, filters)
        x = bottle_neck_block(f"down_{i*2+1}", x, filters, downsample = True)
        filters = int(filters * 2)
        fms.append(x)

    for i in range(len(fms) - 2, -1, -1):
        filters = int(filters // 2)
        x = upsample_block(f"up_{i}", x, fms[i], filters)

    return x


def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    input_tensor = Input(shape=(input_height, input_width, 3))

    x = encoder(8, input_tensor)
    x = bottle_neck_block("semseg_head", x, 8)
    out = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, activation="sigmoid", kernel_regularizer=l2(l=0.0001))(x)

    model = Model(inputs=[input_tensor], outputs=[out])
    return model


if __name__ == "__main__":
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.input.set_shape((1,) + model.input.shape[1:])
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
