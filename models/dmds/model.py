import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from models.dmds import DmdsParams
from common.utils import tflite_convert

class CustomLoss:
    def custom_loss(self, in0, x0, x1, mmap):
        loss = tf.reduce_mean(x0) + tf.reduce_mean(x1) + tf.reduce_mean(mmap)
        return loss

def create_model(input_height: int, input_width: int, base_model_path: str = None) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    in0 = Input(shape=(input_height, input_width, 3))
    in1 = Input(shape=(input_height, input_width, 3))

    x0 = Conv2D(1, (1, 1), padding="same", activation="relu")(in0)
    x1 = Conv2D(1, (1, 1), padding="same", activation="relu")(in1)
    
    x = Concatenate()([in0, in1, x0, x1]) #, in1, x1])
    mmap = Conv2D(1, (1, 1), padding="same", activation="relu")(x)

    model = Model(inputs=[in0, in1], outputs=[x0, x1, mmap])
    custom_loss = CustomLoss()
    model.add_loss(custom_loss.custom_loss(in0, x0, x1, mmap))

    return model


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    # model.input.set_shape((1,) + model.input.shape[1:])
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    # tflite_convert(model, "./tmp", True, True, create_dataset(model.input.shape))
