import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block
from numba.typed import List
from models.semseg import SemsegParams
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from common.utils import to_3channel
from tensorflow.python.keras.engine import compile_utils


def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    inp = Input(shape=(input_height, input_width, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)
    fms = [] # [inp_rescaled]
    namescope = "semseg/"
    filters = 8

    # Downsample
    # ----------------------------
    x = Conv2D(filters, 5, padding="same", name=f"{namescope}initial_downsample", strides=(2, 2), kernel_regularizer=l2(l=0.0001))(inp_rescaled)
    x = BatchNormalization(name=f"{namescope}initial_batchnorm")(x)
    x = ReLU(6., name=f"{namescope}initial_acitvation")(x)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_0/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_1/", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"{namescope}downsample_2/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_3/", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"{namescope}downsample_4/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_5/", x, filters, downsample = True)
    filters = int(filters * 2)

    # Dilation (to avoid further downsampling)
    # ----------------------------
    dilation_rates = [3, 6, 9, 12]
    concat_tensors = []
    x = Conv2D(filters, kernel_size=1, name=f"{namescope}start_dilation_1x1", kernel_regularizer=l2(l=0.0001))(x)
    concat_tensors.append(x)
    for i, rate in enumerate(dilation_rates):
        x = bottle_neck_block(f"{namescope}dilation_{i}", x, filters)
        x = BatchNormalization(name=f"{namescope}dilation_batchnorm_{i}")(x)
        concat_tensors.append(x)

    x = Concatenate(name=f"{namescope}dilation_concat")(concat_tensors)
    fms.append(x)

    # Upsample
    # ----------------------------
    for i in range(len(fms) - 2, -1, -1):
        filters = int(filters // 2)
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
        fms[i] = ReLU(6.0)(fms[i])
        x = upsample_block(f"{namescope}upsample_{i}/", x, fms[i], filters)

    # Create Semseg Map
    # ----------------------------
    x = Conv2D(8, (3, 3), padding="same", name=f"{namescope}semseghead_conv2d", kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization(name=f"{namescope}semseghead_batchnorm")(x)
    x = ReLU(6.0)(x)
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name=f"{namescope}out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(x)

    return Model(inputs=[inp], outputs=semseg_map)


# To test model creation and quickly check if edgetpu compiler compiles it correctly
if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    from models.semseg import convert
    from common.utils import set_weights, tflite_convert
    
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    set_weights.set_weights("/home/computer-vision-models/keras.h5", model, force_resize=False)
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert.tflite_convert(model, "./tmp", True, True, convert.create_dataset(model.input.shape))
