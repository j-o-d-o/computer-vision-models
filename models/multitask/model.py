import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from common.layers import bottle_neck_block, upsample_block, encoder
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.centernet.model import create_output_layer as create_centernet_output_layer
from models.multitask import MultitaskParams


def create_model(params: MultitaskParams) -> tf.keras.Model:
    inp = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)
    
    # Base Model
    # ------------------------------
    namescope = "multitask_base/"
    fms = []
    filter_size = 12
    filters = (np.array([1, 2, 4, 6, 8, 12, 16]) * filter_size)
    x = inp_rescaled

    x = Conv2D(filters[0], 5, padding="same", name=f"{namescope}initial_downsample", strides=(2, 2))(x)
    x = BatchNormalization(name=f"{namescope}initial_batchnorm")(x)
    x = ReLU(6., name=f"{namescope}initial_acitvation")(x)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_2/", x, filters[1])
    x = bottle_neck_block(f"{namescope}downsample_3/", x, filters[1], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_4/", x, filters[2])
    x = bottle_neck_block(f"{namescope}downsample_5/", x, filters[2], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_6/", x, filters[3])
    x = bottle_neck_block(f"{namescope}downsample_7/", x, filters[3], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_8/", x, filters[4])
    x = bottle_neck_block(f"{namescope}downsample_9/", x, filters[4], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_10/", x, filters[5])
    x = bottle_neck_block(f"{namescope}downsample_11/", x, filters[5], downsample = True)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_12/", x, filters[6])
    x = bottle_neck_block(f"{namescope}downsample_13/", x, filters[6], downsample = True)
    fms.append(x)

    filters = filters // 1.5
    filters = filters.astype(np.int32)

    # Semseg Head
    # ------------------------------
    namescope = "semseg/"
    sx = x
    for i in range(len(fms) - 2, -1, -1):
        fms[i] = Conv2D(filters[i], (3, 3), padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
        fms[i] = ReLU(6.0)(fms[i])
        sx = upsample_block(f"{namescope}upsample_{i}/", sx, fms[i], filters[i])

    sx = Conv2D(8, (3, 3), padding="same", name=f"{namescope}semseghead_conv2d", kernel_regularizer=l2(l=0.0001))(sx)
    sx = BatchNormalization(name=f"{namescope}semseghead_batchnorm")(sx)
    sx = ReLU(6.0)(sx)
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name=f"{namescope}out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(sx)

    # Depth Head
    # ------------------------------
    namescope = "depth_model/"
    dx = x
    for i in range(len(fms) - 2, -1, -1):
        fms[i] = Conv2D(filters[i], (3, 3), padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
        fms[i] = ReLU(6.0)(fms[i])
        dx = upsample_block(f"{namescope}upsample_{i}/", dx, fms[i], filters[i])

    dx = Conv2D(6, (3, 3), use_bias=False, padding="same", name=f"{namescope}depthhead_conv2d")(dx)
    dx = BatchNormalization(name=f"{namescope}depthhead_batchnorm")(dx)
    dx = ReLU()(dx)
    depth_map = Conv2D(1, kernel_size=1, padding="same", activation="relu", use_bias=True, name=f"{namescope}out")(dx)

    # Centernet Head
    # ------------------------------
    namescope = "centernet/"
    cx = x
    for i in range(len(fms) - 2, -1, -1):
        fms[i] = Conv2D(filters[i], (3, 3), padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
        fms[i] = ReLU(6.0)(fms[i])
        cx = upsample_block(f"{namescope}upsample_{i}/", cx, fms[i], filters[i])
    centernet_output = create_centernet_output_layer(cx, params.cn_params, namescope)

    # Concatenate (TODO: and refine?)
    # ------------------------------
    out_layer = Concatenate()([centernet_output, semseg_map, depth_map])

    return Model(inputs=[inp], outputs=out_layer, name="multitask/model")


if __name__ == "__main__":
    from data.label_spec import OD_CLASS_MAPPING
    from tensorflow.keras.utils import plot_model
    from models.multitask import create_dataset
    from common.utils import tflite_convert

    params = MultitaskParams(len(OD_CLASS_MAPPING.items()))
    model = create_model(params)
    model.summary()
    plot_model(model, to_file="./tmp/multitask_model.png")
    tflite_convert.tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
