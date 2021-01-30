import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data.semseg_spec import SEMSEG_CLASS_MAPPING


def _bottle_neck_block(inputs: tf.Tensor, filters: int, expansion_factor: int = 6, dilation_rate: int = 1, downsample: bool =False) -> tf.Tensor:
    """
    Bottleneck blocks (As introduced in MobileNet(v2): https://arxiv.org/abs/1801.04381) and extended in functionallity to downsample and to add dilation rates
    :params inputs: Input tensor
    :params filters: Number of filters used
    :params expansion_factor: Factor that multiplies with the number of filters for expansion (1, 1) convolution
    :params dilation_rate: Dialation rate that is used on the Depthwise convolution to increase receptive field
    :params downsample: If True the the layers will be downsampled by 2 with stride 2
    :return: Resulting tensor
    """
    stride = 1 if not downsample else 2
    skip = inputs
    # Expansion
    x = Conv2D(filters * expansion_factor, kernel_size=1, use_bias=False, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    # Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=stride, dilation_rate=dilation_rate, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    # Project
    x = Conv2D(filters, kernel_size=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    # Residual connection
    input_filters = int(inputs.shape[-1])
    if downsample:
        skip = Conv2D(filters, kernel_size=3, strides=2, padding="same")(skip)
    elif input_filters != filters:
        skip = Conv2D(filters, (1, 1), padding="same")(skip)
    x = Add()([skip, x])
    # TODO: Remove these two layers
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def _upsample_block(inputs: tf.Tensor, concat: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Upsample block will upsample one tensor x2 and concatenate with another tensor of the size
    :params inputs: Input tensor to be upsampled by x2
    :params concat: Tensor that will be concatenated, must be x2 size of inputs tensor
    :params filters: Number filters that are used for all convolutions
    :return: Resulting Tensor
    """
    # Upsample input
    x = Conv2DTranspose(filters, kernel_size=2, strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Concatenate
    concat = Conv2D(filters, kernel_size=1)(concat)
    x = Concatenate()([x, concat])
    # Conv
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    inputs = [Input(shape=(input_height, input_width, 3))]

    fs = 12 # filter scaling
    fms = [] # feature maps

    # Feature maps downsampling
    x = _bottle_neck_block(inputs, 1 * fs)
    x = _bottle_neck_block(x, 1 * fs)
    fms.append(x)
    x = _bottle_neck_block(x, 1 * fs, downsample=True)
    x = _bottle_neck_block(x, 2 * fs)
    fms.append(x)

    # Dialated layers to avoid further downsampling while still having large respetive field
    x = _bottle_neck_block(x, 2 * fs, downsample=True)
    d1 = _bottle_neck_block(x, 2 * fs, dilation_rate=3)
    d2 = _bottle_neck_block(x, 2 * fs, dilation_rate=6)
    d3 = _bottle_neck_block(x, 2 * fs, dilation_rate=9)
    d4 = _bottle_neck_block(x, 2 * fs, dilation_rate=12)
    x = Concatenate()([d1, d2, d3, d4])
    x = Conv2D(2 * fs, kernel_size=1, padding="same")(x)
    fms.append(x)

    # Upsampling the blocks to original image size / mask size
    x = _upsample_block(fms[2], fms[1], 2 * fs)
    x = _upsample_block(x, fms[0], 2 * fs)

    out = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, kernel_regularizer=l2(l=0.0001))(x)

    model = Model(inputs=inputs, outputs=[out])
    return model
