import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from models.depth import Params
from models.depth.convert import create_dataset
from common.utils import tflite_convert, resize_img
from common.layers import encoder, upsample_block, bottle_neck_block
from common.utils.set_weights import set_weights


def create_model(input_height: int, input_width: int, base_model_path: str = None) -> tf.keras.Model:
    inp = Input(shape=(input_height, input_width, 3))
    x0, _ = encoder(8, inp, namescope="depth_model")
    x0 = Conv2D(16, (3, 3), padding="same", name="depth_model_conv2d")(x0)
    x0 = ReLU()(x0)
    x0 = BatchNormalization()(x0)
    x0 = Conv2D(1, kernel_size=1, padding="same", activation="softplus", name="depth_map")(x0)

    depth_model = Model(inputs=inp, outputs=x0)
    if base_model_path is not None:
        depth_model = set_weights(base_model_path, depth_model)

    return depth_model


if __name__ == "__main__":
    params = Params()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/depth_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))
