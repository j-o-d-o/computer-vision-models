from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, ZeroPadding2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.regularizers import l2
from data.semseg_spec import SEMSEG_CLASS_MAPPING
from models.semseg.params import Params


def downsample_block(inputs, filters: int, kernel=(3, 3)):
    conv1 = Conv2D(filters, kernel, padding='same', kernel_regularizer=l2(l=0.0001))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(filters, kernel, padding='same', kernel_regularizer=l2(l=0.0001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    return pool, conv2


def upsample_bock(inputs, concat_layer, filters: int, kernel=(3, 3), final_layer: bool = False):
    up_layer = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)

    # find crop values for height
    pad_up_layer = up_layer.shape[1] < concat_layer.shape[1]
    diff_height = abs(up_layer.shape[1] - concat_layer.shape[1])
    add_uneven_height = 0
    if diff_height % 2 != 0:
        add_uneven_height = 1
        diff_height -= 1 # make it even
    diff_height_tuple = (int(diff_height * 0.5) + add_uneven_height, int(diff_height * 0.5))
    # find crop values for width
    diff_width = abs(up_layer.shape[2] - concat_layer.shape[2])
    add_uneven_width = 0
    if diff_width % 2 != 0:
        add_uneven_width = 1
        diff_width -= 1 # make it even
    diff_width_tuple = (int(diff_width * 0.5), int(diff_width * 0.5) + add_uneven_width)

    if pad_up_layer:
        padded_up_layer = ZeroPadding2D((diff_height_tuple, diff_width_tuple))(up_layer)
        up = concatenate([padded_up_layer, concat_layer], axis=3)
    else:
        padded_concat_layer = ZeroPadding2D((diff_height_tuple, diff_width_tuple))(concat_layer)
        up = concatenate([up_layer, padded_concat_layer], axis=3)

    conv1 = Conv2D(filters*2, kernel, padding='same')(up)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if final_layer:
        conv2 = Conv2D(filters, kernel, padding='same', kernel_regularizer=l2(l=0.0001))(conv1)
    else:
        conv2 = Conv2D(filters, kernel, padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return conv2


def quantize_model(model):
    def apply_quantization_annotation(layer):
        # Add all layers to the tuple that currently do not have any quantization support
        if not isinstance(layer, (Conv2DTranspose)):
            return quantize_annotate_layer(layer)
        return layer

    annotated_model = clone_model(model, clone_function=apply_quantization_annotation)
    quant_aware_model = quantize_apply(annotated_model)
    return quant_aware_model

def create_model():
    inputs = Input(shape=(Params.INPUT_HEIGHT, Params.INPUT_WIDTH, Params.INTPUT_CHANNELS))

    pool1, conv_down_1 = downsample_block(inputs, 32, kernel=(5, 5))
    pool2, conv_down_2 = downsample_block(pool1, 48)
    pool3, conv_down_3 = downsample_block(pool2, 64)
    pool4, conv_down_4 = downsample_block(pool3, 128)
    pool5, conv_down_5 = downsample_block(pool4, 254)
    pool5 = Dropout(0.4)(pool5)

    conv = Conv2D(254, (3, 3), padding='same')(pool5)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(0.4)(conv)

    conv_up = upsample_bock(conv,    conv_down_5, 254)
    conv_up = Dropout(0.4)(conv_up)
    conv_up = upsample_bock(conv_up, conv_down_4, 128)
    conv_up = Dropout(0.4)(conv_up)
    conv_up = upsample_bock(conv_up, conv_down_3, 64)
    conv_up = Dropout(0.35)(conv_up)
    conv_up = upsample_bock(conv_up, conv_down_2, 32)
    conv_up = Dropout(0.3)(conv_up)
    conv_up = upsample_bock(conv_up, conv_down_1, 16, final_layer=True)
    conv_up = Dropout(0.25)(conv_up)

    out = Conv2D(len(SEMSEG_CLASS_MAPPING), (1, 1), kernel_regularizer=l2(l=0.0001))(conv_up)

    model = Model(inputs=[inputs], outputs=[out])
    return model
