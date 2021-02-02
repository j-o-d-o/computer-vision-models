import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data.semseg_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block


def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    input_layer = Input(shape=(input_height, input_width, 3))

    fs = 12 # filter scaling
    fms = [] # feature maps

    # Feature maps downsampling
    x = bottle_neck_block(input_layer, 1 * fs)
    x = bottle_neck_block(x, 1 * fs)
    fms.append(x)
    x = bottle_neck_block(x, 1 * fs, downsample=True)
    x = bottle_neck_block(x, 2 * fs)
    fms.append(x)

    # Dialated layers to avoid further downsampling while still having large respetive field
    x = bottle_neck_block(x, 2 * fs, downsample=True)
    d1 = bottle_neck_block(x, 2 * fs, dilation_rate=3)
    d2 = bottle_neck_block(x, 2 * fs, dilation_rate=6)
    d3 = bottle_neck_block(x, 2 * fs, dilation_rate=9)
    d4 = bottle_neck_block(x, 2 * fs, dilation_rate=12)
    x = Concatenate()([d1, d2, d3, d4])
    x = Conv2D(2 * fs, kernel_size=1, padding="same")(x)
    fms.append(x)

    # Upsampling the blocks to original image size / mask size
    x = upsample_block(fms[2], fms[1], 2 * fs)
    x = upsample_block(x, fms[0], 2 * fs)

    out = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, kernel_regularizer=l2(l=0.0001))(x)

    model = Model(inputs=input_layer, outputs=[out])
    return model

# params = SemsegParams()
# model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
# model.summary()
