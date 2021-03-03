import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.initializers import Constant, GlorotNormal
from tensorflow.keras.regularizers import l2


def bottle_neck_block(name_prefix: str, inputs: tf.Tensor, filters: int, expansion_factor: int = 6, dilation_rate: int = 1, downsample: bool =False) -> tf.Tensor:
    """
    Bottleneck blocks (As introduced in MobileNet(v2): https://arxiv.org/abs/1801.04381) and extended in functionallity to downsample and to add dilation rates
    """
    stride = 1 if not downsample else 2
    skip = inputs

    # # Expansion
    # x = Conv2D(filters * expansion_factor, kernel_size=1, use_bias=False,
    #     padding='same', name=f"{name_prefix}conv2d/0", kernel_regularizer=l2(l=0.0001))(inputs)
    # x = BatchNormalization(name=f"{name_prefix}batchnorm/0")(x)
    # x = ReLU(6.0)(x)
    # # Convolution
    # x = DepthwiseConv2D(kernel_size=3, strides=stride, dilation_rate=dilation_rate,
    #     use_bias=False, padding='same', name=f"{name_prefix}conv2d/1", kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization(name=f"{name_prefix}batchnorm/1")(x)
    # x = ReLU(6.0)(x)
    # # Project
    # x = Conv2D(filters, kernel_size=1, use_bias=False, padding='same', name=f"{name_prefix}conv2d/2", kernel_regularizer=l2(l=0.0001))(x)
    # x = BatchNormalization(name=f"{name_prefix}batchnorm/2")(x)

    x = Conv2D(filters, use_bias=False, kernel_size=3, name=f"{name_prefix}conv2d/0", padding="same", dilation_rate=dilation_rate, kernel_regularizer=l2(l=0.0001))(inputs)
    x = BatchNormalization(name=f"{name_prefix}batchnorm/0")(x)
    x = ReLU(6.0)(x)
    
    x = Conv2D(filters, use_bias=False, kernel_size=3, name=f"{name_prefix}conv2d/1", padding="same", strides=stride, dilation_rate=dilation_rate, kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization(name=f"{name_prefix}batchnorm/1")(x)
    x = ReLU(6.0)(x)

    # Residual connection
    input_filters = int(inputs.shape[-1])
    if downsample:
        skip = Conv2D(filters, kernel_size=3, strides=2, padding="same", name=f"{name_prefix}conv2d/3", kernel_regularizer=l2(l=0.0001))(skip)
    elif input_filters != filters:
        skip = Conv2D(filters, (1, 1), padding="same", name=f"{name_prefix}skip", kernel_regularizer=l2(l=0.0001))(skip)
    x = Add(name=f"{name_prefix}add")([skip, x])
    x = BatchNormalization(name=f"{name_prefix}batchnorm/3")(x)
    x = ReLU(6.0, name=f"{name_prefix}out")(x)
    return x


def upsample_block(name_prefix: str, inputs: tf.Tensor, concat: tf.Tensor, filters: int) -> tf.Tensor:
    """
    Upsample block will upsample one tensor x2 and concatenate with another tensor of the size
    """
    # Upsample inputs
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), use_bias=False, padding='same', name=f"{name_prefix}conv2dtranspose", kernel_regularizer=l2(l=0.0001))(inputs)
    x = BatchNormalization(name=f"{name_prefix}batchnorm/0")(x)
    x = ReLU(6.0)(x)
    # Concatenate
    concat = Conv2D(filters, use_bias=False, kernel_size=1, name=f"{name_prefix}conv2d/1", kernel_regularizer=l2(l=0.0001))(concat)
    concat = BatchNormalization(name=f"{name_prefix}batchnorm/1")(concat)
    concat = ReLU(6.0)(concat)
    x = Concatenate()([x, concat])
    # Conv
    x = Conv2D(filters, kernel_size=3, use_bias=False, padding='same', name=f"{name_prefix}conv2d/2", kernel_regularizer=l2(l=0.0001))(x)
    x = BatchNormalization(name=f"{name_prefix}batchnorm/2")(x)
    x = ReLU(6.0, name=f"{name_prefix}out")(x)
    return x
