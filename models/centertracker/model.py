# Check: https://coral.ai/docs/edgetpu/models-intro/#supported-operations for supported ops on EdgeTpu
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, Model
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Conv2D, DepthwiseConv2D, Add, Input, Concatenate
import numpy as np
from kerassurgeon.operations import insert_layer
from models.centertracker.params import CentertrackerParams
import models.centernet as centernet


def create_model(params: CentertrackerParams):
    base_model = centernet.create_model(params)
    curr_img_input = base_model.inputs[0]
    prev_img_input = Input(shape=(params.INPUT_HEIGHT, params.INPUT_WIDTH, 3))
    prev_heatmap_input = Input(shape=(params.MASK_HEIGHT, params.MASK_WIDTH, 1))

    # rewire the added image from t-1 to the second layer of the base_model
    img_input = Concatenate(axis=3)([curr_img_input, prev_img_input])
    base_model.layers[1].input = img_input

    base_feature = base_model.get_layer("encoder_output").output
    output_layer = base_model.output

    # All other regerssion parameters are optional, but note that the order is important here and should be as in the OrderedDict REGRESSION_FIELDS
    if params.REGRESSION_FIELDS["track_offset"].active:
        # Create location offset due to R scaling
        track_offset = Conv2D(32, (3, 3), padding="same", use_bias=False)(base_feature)
        track_offset = BatchNormalization()(track_offset)
        track_offset = ReLU()(track_offset)
        track_offset = Conv2D(params.REGRESSION_FIELDS["track_offset"].size, (1, 1), padding="valid", activation=None)(track_offset)
        output_layer = Concatenate(axis=3)([output_layer, track_offset])

    # Create Model
    input_dict = {
        "img": curr_img_input,
        "prev_img": prev_img_input,
        "prev_heatmap": prev_heatmap_input
    }
    model = Model(inputs=input_dict, outputs=output_layer, name="centernet")
    model.summary()
    return model
