# Check: https://coral.ai/docs/edgetpu/models-intro/#supported-operations for supported ops on EdgeTpu
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, Model
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, Input, Concatenate
import numpy as np
from models.centernet.params import Params


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
    # TODO: Remove these two layers?
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
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
    x = ReLU()(x)
    return x

def create_model(params: Params):
    fs = 16 # filter scaling
    fms = [] # feature maps

    inputs = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3))
    x = Conv2D(max(2, int(fs)), 5, padding="same", strides=(2, 2))(inputs) # / 2
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    # Downsample
    x = _bottle_neck_block(x, 1 * fs)
    x = _bottle_neck_block(x, 1 * fs)
    fms.append(x)
    x = _bottle_neck_block(x, 1 * fs, downsample=True)
    x = _bottle_neck_block(x, 2 * fs)
    fms.append(x)

    x = _bottle_neck_block(x, 2 * fs, downsample=True)
    d1 = _bottle_neck_block(x, 2 * fs, dilation_rate=3)
    d2 = _bottle_neck_block(x, 2 * fs, dilation_rate=6)
    d3 = _bottle_neck_block(x, 2 * fs, dilation_rate=9)
    d4 = _bottle_neck_block(x, 2 * fs, dilation_rate=12)
    x = Concatenate()([d1, d2, d3, d4])
    x = Conv2D(2 * fs, kernel_size=1, padding="same")(x)
    fms.append(x)

    # features sizes with input (144, 608)
    # [0]: (None, 304, 72, x)
    # [1]: (None, 152, 36, x)
    # [2]: (None, 76, 18, x)
    # -> with R = 2, we need 2 upsamples

    x = _upsample_block(fms[2], fms[1], 2 * fs)
    x = _upsample_block(x, fms[0], 2 * fs)

    output_layer_arr = []

    # Create heatmap / class output
    heatmap = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    heatmap = BatchNormalization()(heatmap)
    heatmap = ReLU()(heatmap)
    heatmap = Conv2D(params.NB_CLASSES, (1, 1), padding="valid", activation=tf.nn.sigmoid)(heatmap)
    output_layer_arr.append(heatmap)

    # All other regerssion parameters are optional, but note that the order is important here and should be as in the OrderedDict REGRESSION_FIELDS
    if params.REGRESSION_FIELDS["r_offset"].active:
        # Create location offset due to R scaling
        offset = Conv2D(16, (3, 3), padding="same", use_bias=False)(x)
        offset = BatchNormalization()(offset)
        offset = ReLU()(offset)
        offset = Conv2D(params.REGRESSION_FIELDS["r_offset"].size, (1, 1), padding="valid", activation=None)(offset)
        output_layer_arr.append(offset)

    if params.REGRESSION_FIELDS["fullbox"].active:
        # Create 2D output: fullbox (width, height)
        fullbox = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
        fullbox = BatchNormalization()(fullbox)
        fullbox = ReLU()(fullbox)
        fullbox = Conv2D(params.REGRESSION_FIELDS["fullbox"].size, (1, 1), padding="valid", activation=None)(fullbox)
        output_layer_arr.append(fullbox)

    if params.REGRESSION_FIELDS["l_shape"].active:
        # Create 2.5D output: bottom_left_edge_point, bottom_right_edge_point, bottom_center_edge_point, center_height
        l_shape = Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
        l_shape = BatchNormalization()(l_shape)
        l_shape = ReLU()(l_shape)
        l_shape = Conv2D(params.REGRESSION_FIELDS["l_shape"].size, (1, 1), padding="valid", activation=None)(l_shape)
        output_layer_arr.append(l_shape)

    if params.REGRESSION_FIELDS["3d_info"].active:
        # Create radial distance output
        radial_dist = Conv2D(16, (3, 3), padding="same", use_bias=False)(x)
        radial_dist = BatchNormalization()(radial_dist)
        radial_dist = ReLU()(radial_dist)
        radial_dist = Conv2D(1, (1, 1), padding="valid", activation=None)(radial_dist)
        output_layer_arr.append(radial_dist)

        # Create orientation output
        orientation = Conv2D(16, (3, 3), padding="same", use_bias=False)(x)
        orientation = BatchNormalization()(orientation)
        orientation = ReLU()(orientation)
        orientation = Conv2D(1, (1, 1), padding="valid", activation=None)(orientation)
        output_layer_arr.append(orientation)

        # Create object dimensions in [m] (width, height, length)
        obj_dims = Conv2D(16, (3, 3), padding="same", use_bias=False)(x)
        obj_dims = BatchNormalization()(obj_dims)
        obj_dims = ReLU()(obj_dims)
        obj_dims = Conv2D(3, (1, 1), padding="valid", activation=None)(obj_dims)
        output_layer_arr.append(obj_dims)

    # Concatenate output
    output_layer = Concatenate(axis=3)(output_layer_arr)

    # Create Model
    model = Model(inputs=inputs, outputs=output_layer, name="centernet")

    return model
