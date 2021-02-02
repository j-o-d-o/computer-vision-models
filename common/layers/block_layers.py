import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, Conv2DTranspose, Concatenate


def bottle_neck_block(inputs: tf.Tensor, filters: int, expansion_factor: int = 6, dilation_rate: int = 1, downsample: bool =False, name: str = None) -> tf.Tensor:
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
    x = Add(name=name)([skip, x])
    # TODO: Remove these two layers?
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    return x


def upsample_block(inputs: tf.Tensor, concat: tf.Tensor, filters: int, name: str = None) -> tf.Tensor:
    """
    Upsample block will upsample one tensor x2 and concatenate with another tensor of the size
    :params inputs: Input tensor to be upsampled by x2
    :params concat: Tensor that will be concatenated, must be x2 size of inputs tensor
    :params filters: Number filters that are used for all convolutions
    :return: Resulting Tensor
    """
    # Upsample input
    x = Conv2DTranspose(filters, kernel_size=2, strides=(2, 2), use_bias=False, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # Concatenate
    concat = Conv2D(filters, use_bias=False, kernel_size=1)(concat)
    concat = BatchNormalization()(concat)
    concat = ReLU()(concat)
    x = Concatenate()([x, concat])
    # Conv
    x = Conv2D(filters, kernel_size=3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(name=name)(x)
    return x
