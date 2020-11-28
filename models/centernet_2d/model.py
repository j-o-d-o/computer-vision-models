import tensorflow as tf
import numpy as np
import tensorflow.python.keras.applications.efficientnet as efficientnet
from models.centernet_2d import Params


def deConv2DBatchNorm(inputs, filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", name=None):
    """
    Deconvolution with BatchNormalization and activation
    :param inputs: input to the layer
    :param filters: number of filters from convoltuion, defaults to 128
    :param kernel_size: kernel for convolution, defaults to (2, 2)
    :param strides: strides of convolution, defaults to (2, 2)
    :param name: name of the scope
    :return: output deconvolution layer
    """
    layer = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1.25e-5),
        name=name + "_deconv2D"
    )(inputs)
    layer = tf.keras.layers.BatchNormalization(name=name + "_batnchnorm")(layer)
    layer = tf.keras.layers.Activation("relu", name=name + "_activation")(layer)
    return layer

def create_model(input_height, input_width, mask_height, mask_width, nb_classes):
    """
    Using some pretrained classifier model on imagenet from the keras application zoo as base,
    add the needed layers to get to the output mask size
    """
    shape = (input_height, input_width, 3)
    base_model = efficientnet.EfficientNetB2(input_shape=shape, include_top=False, weights='imagenet')

    # Extract feature layers and upsample them to the output shape
    output_shape = (mask_height, mask_width)
    # block2c_activation
    feat_0 = deConv2DBatchNorm(base_model.get_layer("block2c_activation").output, filters=16, name="up0_feat_0")
    feat_0 = tf.raw_ops.ResizeNearestNeighbor(images=feat_0, size=output_shape, name="feat_0_resize")
    # block3c_activation
    feat_1 = deConv2DBatchNorm(base_model.get_layer("block3c_activation").output, filters=16, name="up0_feat_1")
    feat_1 = deConv2DBatchNorm(feat_1, filters=8, name="up1_feat_1")
    feat_1 = tf.raw_ops.ResizeNearestNeighbor(images=feat_1, size=output_shape, name="feat_1_resize")
    # block5d_activation
    feat_2 = deConv2DBatchNorm(base_model.get_layer("block5d_activation").output, filters=32, name="up0_feat_2")
    feat_2 = deConv2DBatchNorm(feat_2, filters=16, name="up1_feat_2")
    feat_2 = deConv2DBatchNorm(feat_2, filters=8, name="up2_feat_2")
    feat_2 = tf.raw_ops.ResizeNearestNeighbor(images=feat_2, size=output_shape, name="feat_2_resize")
    # top_activation
    feat_3 = deConv2DBatchNorm(base_model.get_layer("top_activation").output, filters=64, name="up0_feat_3")
    feat_3 = deConv2DBatchNorm(feat_3, filters=32, name="up1_feat_3")
    feat_3 = deConv2DBatchNorm(feat_3, filters=16, name="up2_feat_3")
    feat_3 = deConv2DBatchNorm(feat_3, filters=8, name="up3_feat_3")
    feat_3 = tf.raw_ops.ResizeNearestNeighbor(images=feat_3, size=output_shape, name="feat_3_resize")

    features = tf.keras.layers.concatenate([feat_0, feat_1, feat_2, feat_3], axis=3)

    # Create heatmap / class output
    output_heatmap = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.01), kernel_regularizer=tf.keras.regularizers.l2(1.25e-5),
        name="heatmap_conv2D")(features)
    output_heatmap = tf.keras.layers.BatchNormalization(name="heatmap_norm")(output_heatmap)
    output_heatmap = tf.keras.layers.Activation("relu", name="heatmap_activ")(output_heatmap)
    output_heatmap = tf.keras.layers.Conv2D(nb_classes, (1, 1), padding="valid", activation=tf.nn.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.01), kernel_regularizer=tf.keras.regularizers.l2(1.25e-5),
        bias_initializer=tf.constant_initializer(-np.log((1.0 - 0.1) / 0.1)), name="heatmap")(output_heatmap)
    
    # Create size output
    output_bbox_size = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.001), kernel_regularizer=tf.keras.regularizers.l2(1.25e-5),
        name="size_conv2D")(features)
    output_bbox_size = tf.keras.layers.BatchNormalization(name="size_norm")(output_bbox_size)
    output_bbox_size = tf.keras.layers.Activation("relu", name="size_activ")(output_bbox_size)
    output_bbox_size = tf.keras.layers.Conv2D(2, (1, 1), padding="valid", activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.001), kernel_regularizer=tf.keras.regularizers.l2(1.25e-5),
        name="bounding_box_size")(output_bbox_size)

    # Concatenate output
    output_layer = tf.keras.layers.concatenate([output_heatmap, output_bbox_size], axis=3)

    # Create Model
    model = tf.keras.Model(inputs=base_model.input, outputs=output_layer, name="centernet2d")

    return model
