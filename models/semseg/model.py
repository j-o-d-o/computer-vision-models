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
    x = input_tensor
    fms = []

    x = Conv2D(filters, 5, padding="same", name="initial_downsample", strides=(2, 2))(x) # / 2
    x = BatchNormalization(name="initial_batchnorm")(x)
    x = ReLU(6., name="initial_acitvation")(x)
    fms.append(x)

    x = bottle_neck_block(f"feat_extract_0", x, filters)
    x = bottle_neck_block(f"feat_extract_1", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"feat_extract_2", x, filters)
    x = bottle_neck_block(f"feat_extract_2.5", x, filters)
    x = bottle_neck_block(f"feat_extract_3", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"feat_extract_4", x, filters)
    x = bottle_neck_block(f"feat_extract_4.5", x, filters)
    x = bottle_neck_block(f"feat_extract_5", x, filters, downsample = True)
    filters = int(filters * 2)

    dilation_rates = [3, 5, 7, 9, 11]
    concat_tensors = []
    x = Conv2D(filters, kernel_size=1, name="start_dilation_1x1")(x)
    concat_tensors.append(x)
    for i, rate in enumerate(dilation_rates):
        x = bottle_neck_block(f"dilation_{i}", x, filters)
        x = BatchNormalization(name=f"dilation_batchnorm_{i}")(x)
        concat_tensors.append(x)

    x = Concatenate(name="dilation_concat")(concat_tensors)
    fms.append(x)

    for i in range(len(fms) - 2, -1, -1):
        filters = int(filters // 2)
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"conv2d_up1_{i}")(fms[i])
        fms[i] = BatchNormalization(name=f"batchnorm_up1_{i}")(fms[i])
        fms[i] = ReLU()(fms[i])
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"conv2d_up2_{i}")(fms[i])
        fms[i] = BatchNormalization(name=f"batchnorm_up2_{i}")(fms[i])
        fms[i] = ReLU()(fms[i])
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

    x = encoder(16, input_tensor)

    x = Conv2D(32, (3, 3), padding="same", name="semseg_head_conv2d", use_bias=False)(x)
    x = BatchNormalization(name="semseg_head_batchnorm")(x)
    x = ReLU(name="semseg_head_activation")(x)

    out = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg_out", kernel_regularizer=l2(l=0.0001))(x)

    model = Model(inputs=[input_tensor], outputs=[out])
    return model


if __name__ == "__main__":
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.input.set_shape((1,) + model.input.shape[1:])
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
