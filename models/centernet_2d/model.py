# Check: https://coral.ai/docs/edgetpu/models-intro/#supported-operations for supported ops on EdgeTpu
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, Model
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, Input, concatenate
from tensorflow.raw_ops import ResizeNearestNeighbor
import numpy as np


def deconv_2d(inputs, filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    layer = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False,
        kernel_initializer="he_uniform", kernel_regularizer=regularizers.l2(1.25e-5))(inputs)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    return layer

def _bottle_neck_block(inputs, filters: int, expansion_factor: int = 6, stride=1, bn_alpha=1):
  filters = max(2, int(filters * bn_alpha))
  x = inputs

  # Expansion
  x = Conv2D(filters * expansion_factor, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  # Convolution
  x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                      padding='same' if stride == 1 else 'same')(x)
  x = BatchNormalization()(x)
  x = ReLU(6.)(x)

  # Project
  x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
  x = BatchNormalization()(x)

  # Residual connection
  input_filters = int(inputs.shape[-1])
  if stride == 1 and input_filters == filters:
    x = Add()([inputs, x])

  return x

def create_model(input_height, input_width, mask_height, mask_width, nb_classes):
    bn_alpha = 1
    inputs = Input(shape=(input_height, input_width, 3))
    x = Conv2D(max(2, int(32 * bn_alpha)), 5, padding="same", strides=(2, 2))(inputs) # / 2
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    features = []
    features.append(x)

    x = _bottle_neck_block(x, 32, expansion_factor=1, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 32, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 32, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 32, expansion_factor=6, stride=2, bn_alpha=bn_alpha) # / 2
    features.append(x)

    x = _bottle_neck_block(x, 64, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 64, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 64, expansion_factor=6, stride=2, bn_alpha=bn_alpha) # / 2
    features.append(x)

    x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 98, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 98, expansion_factor=6, stride=2, bn_alpha=bn_alpha) # / 2
    features.append(x)

    x = _bottle_neck_block(x, 128, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 128, expansion_factor=6, stride=1, bn_alpha=bn_alpha)
    x = _bottle_neck_block(x, 128, expansion_factor=6, stride=2, bn_alpha=bn_alpha) # / 2
    features.append(x)

    # features sizes with input (92, 308)
    # [0]: (None, 46, 154, 32)
    # [1]: (None, 23, 77, 32)
    # [2]: (None, 12, 39, 64)
    # [3]: (None, 6, 20, 98)
    # [4]: (None, 3, 10, 128)

    # Extract feature layers and upsample them to the output shape
    output_shape = (mask_height, mask_width)
    # features[0]
    features[0] = tf.raw_ops.ResizeNearestNeighbor(images=features[0], size=output_shape)
    # features[1]
    features[1] = deconv_2d(features[1], filters=16)
    features[1] = tf.raw_ops.ResizeNearestNeighbor(images=features[1], size=output_shape)
    # features[2]
    features[2] = deconv_2d(features[2], filters=32)
    features[2] = deconv_2d(features[2], filters=16)
    features[2] = tf.raw_ops.ResizeNearestNeighbor(images=features[2], size=output_shape)
    # features[3]
    features[3] = deconv_2d(features[3], filters=64)
    features[3] = deconv_2d(features[3], filters=32)
    features[3] = deconv_2d(features[3], filters=16)
    features[3] = tf.raw_ops.ResizeNearestNeighbor(images=features[3], size=output_shape)
    # features[4]
    features[4] = deconv_2d(features[4], filters=128)
    features[4] = deconv_2d(features[4], filters=64)
    features[4] = deconv_2d(features[4], filters=32)
    features[4] = deconv_2d(features[4], filters=16)
    features[4] = tf.raw_ops.ResizeNearestNeighbor(images=features[4], size=output_shape)

    features_layer = concatenate([features[0], features[1], features[2], features[3], features[4]], axis=3)

    # Create heatmap / class output
    output_heatmap = Conv2D(32, (3, 3), padding="same", use_bias=False,
        kernel_initializer=initializers.RandomNormal(0.01), kernel_regularizer=regularizers.l2(1.25e-5),
        name="heatmap_conv2D")(features_layer)
    output_heatmap = BatchNormalization(name="heatmap_norm")(output_heatmap)
    output_heatmap = ReLU(name="heatmap_activ")(output_heatmap)
    output_heatmap = Conv2D(nb_classes, (1, 1), padding="valid", activation=tf.nn.sigmoid,
        kernel_initializer=initializers.RandomNormal(0.01), kernel_regularizer=regularizers.l2(1.25e-5),
        bias_initializer=tf.constant_initializer(-np.log((1.0 - 0.1) / 0.1)), name="heatmap")(output_heatmap)
    
    # Create offset output
    output_offset = Conv2D(32, (3, 3), padding="same", use_bias=False,
        kernel_initializer=initializers.RandomNormal(0.001), kernel_regularizer=regularizers.l2(1.25e-5),
        name="offset_conv2D")(features_layer)
    output_offset = BatchNormalization(name="offset_norm")(output_offset)
    output_offset = ReLU(name="offset_activ")(output_offset)
    output_offset = Conv2D(2, (1, 1), padding="valid", activation=None,
        kernel_initializer=initializers.RandomNormal(0.001), kernel_regularizer=regularizers.l2(1.25e-5),
        name="offset")(output_offset)

    # Create size output
    output_bbox_size = Conv2D(32, (3, 3), padding="same", use_bias=False,
        kernel_initializer=initializers.RandomNormal(0.001), kernel_regularizer=regularizers.l2(1.25e-5),
        name="size_conv2D")(features_layer)
    output_bbox_size = BatchNormalization(name="size_norm")(output_bbox_size)
    output_bbox_size = ReLU(name="size_activ")(output_bbox_size)
    output_bbox_size = Conv2D(2, (1, 1), padding="valid", activation=None,
        kernel_initializer=initializers.RandomNormal(0.001), kernel_regularizer=regularizers.l2(1.25e-5),
        name="bounding_box_size")(output_bbox_size)

    # Concatenate output, order is important here!
    output_layer = concatenate([output_heatmap, output_offset, output_bbox_size], axis=3)

    # Create Model
    model = Model(inputs=inputs, outputs=output_layer, name="centernet2d")

    return model
