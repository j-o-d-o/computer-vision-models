import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, UpSampling2D, Concatenate, Conv2DTranspose
from tensorflow.keras.initializers import Constant, GlorotNormal


def bottle_neck_block(name_prefix: str, inputs: tf.Tensor, filters: int, expansion_factor: int = 6, dilation_rate: int = 1, downsample: bool =False) -> tf.Tensor:
    """
    Bottleneck blocks (As introduced in MobileNet(v2): https://arxiv.org/abs/1801.04381) and extended in functionallity to downsample and to add dilation rates
    """
    stride = 1 if not downsample else 2
    skip = inputs
    # Expansion
    x = Conv2D(filters * expansion_factor, kernel_size=1, use_bias=False,
        padding='same', name=f"conv2d-0_bottelneck_{name_prefix}")(inputs)
    x = BatchNormalization(name=f"batchnorm-0_bottelneck_{name_prefix}")(x)
    x = ReLU()(x)
    # Convolution
    x = DepthwiseConv2D(kernel_size=3, strides=stride, dilation_rate=dilation_rate,
        use_bias=False, padding='same', name=f"conv2d-1_bottelneck_{name_prefix}")(x)
    x = BatchNormalization(name=f"batchnorm-1_bottelneck_{name_prefix}")(x)
    x = ReLU()(x)
    # Project
    x = Conv2D(filters, kernel_size=1, use_bias=False, padding='same', name=f"conv2d-2_bottelneck_{name_prefix}")(x)
    x = BatchNormalization(name=f"batchnorm-2_bottelneck_{name_prefix}")(x)
    # Residual connection
    input_filters = int(inputs.shape[-1])
    if downsample:
        skip = Conv2D(filters, kernel_size=3, strides=2, padding="same", name=f"conv2d-3_bottelneck_{name_prefix}")(skip)
    elif input_filters != filters:
        skip = Conv2D(filters, (1, 1), padding="same", name=f"skip_bottelneck_{name_prefix}")(skip)
    x = Add(name=f"bottelneck_out_{name_prefix}")([skip, x])
    return x


def upsample_block(name_prefix: str, inputs: tf.Tensor, concat: tf.Tensor, filters: int, name: str = None) -> tf.Tensor:
    """
    Upsample block will upsample one tensor x2 and concatenate with another tensor of the size
    """
    # Upsample inputs
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), use_bias=False, padding='same')(inputs)
    x = BatchNormalization(name=f"batchnorm-0_upsample_{name_prefix}")(x)
    x = ReLU()(x)
    # Concatenate
    concat = Conv2D(filters, use_bias=False, kernel_size=1, name=f"conv2d-1_upsample_{name_prefix}")(concat)
    concat = BatchNormalization(name=f"batchnorm-1_upsample_{name_prefix}")(concat)
    concat = ReLU()(concat)
    x = Concatenate()([x, concat])
    # Conv
    x = Conv2D(filters, kernel_size=3, use_bias=False, padding='same', name=f"conv2d-2_upsample_{name_prefix}")(x)
    x = BatchNormalization(name=f"batchnorm-2_upsample_{name_prefix}")(x)
    x = ReLU(name=f"upsample_out_{name_prefix}")(x)
    return x
