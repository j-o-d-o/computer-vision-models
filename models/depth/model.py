import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Constant
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block, encoder
from typing import List
from tensorflow.keras.utils import plot_model
from models.semseg import SemsegParams, create_dataset
from common.utils import tflite_convert


def create_model(input_height: int, input_width: int, base_model_path: str = None) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    input_tensor = Input(shape=(input_height, input_width, 3))

    x = encoder(16, input_tensor)
    encoder_model = Model(inputs=[input_tensor], outputs=[x])
    if base_model_path is not None:
        # Store names of base model layerslayers of model in dict
        base_model = tf.keras.models.load_model(base_model_path, compile=False)
        base_layer_dict = dict([(layer.name, layer) for layer in base_model.layers]) 
        # Loop through actual model and see if names are matching, set weights in case they are
        for layer in encoder_model.layers:
            if layer.name in base_layer_dict:
                print(f"Setting weights for {layer.name}")
                layer.set_weights(base_layer_dict[layer.name].get_weights())
            else:
                print(f"Not found: {layer.name}")


    x = Conv2D(32, (3, 3), padding="same", name="semseg_head_conv2d", use_bias=False)(x)
    x = BatchNormalization(name="semseg_head_batchnorm")(x)
    x = ReLU(name="semseg_head_activation")(x)

    out = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg_out", kernel_regularizer=l2(l=0.0001))(x)

    model = Model(inputs=encoder_model.inputs, outputs=[out])

    return model


if __name__ == "__main__":
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, "/home/computer-vision-models/tmp/load_from_model/keras.h5")
    # model.input.set_shape((1,) + model.input.shape[1:])
    # model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    # tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
