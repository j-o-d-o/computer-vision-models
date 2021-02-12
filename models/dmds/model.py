import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.engine import data_adapter
from models.dmds import DmdsParams
from models.dmds.convert import create_dataset
from common.utils import tflite_convert, resize_img
from common.layers import encoder, upsample_block, bottle_neck_block


# Creating custom train_step() in order to call loss function with all needed parameters
class DmdsModel(Model):
    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        in_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(in_data, training=True)
            img0 = in_data[0]
            img1 = in_data[1]
            intr = in_data[2]
            x0 = y_pred[0]
            x1 = y_pred[1]
            mm = y_pred[2]
            rot = y_pred[3]
            tran = y_pred[4]
            loss = self.custom_loss.calc(img0, img1, x0, x1, mm, rot, tran, intr)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_dict = self.custom_loss.loss_vals
        loss_dict["sum"] = loss
        return loss_dict


def create_model(input_height: int, input_width: int, base_model_path: str = None) -> tf.keras.Model:
    # Depth Maps
    # =======================================
    intr = Input(shape=(3, 3))
    input_t0 = Input(shape=(input_height, input_width, 3))
    input_t1 = Input(shape=(input_height, input_width, 3))
    # x0, _ = encoder(2, input_t0, namescope="depth_t0")
    # x1, _ = encoder(2, input_t1, namescope="depth_t1")
    # depth_model_t0 = Model(inputs=input_t0, outputs=x0)
    # depth_model_t1 = Model(inputs=input_t1, outputs=x1)

    x0 = input_t0
    x1 = input_t1

    x0 = Conv2D(2, (3, 3), padding="same", name="depth_map_t0_conv2d", use_bias=False)(x0)
    x0 = BatchNormalization(name="depth_map_t0_batchnorm")(x0)
    x0 = ReLU()(x0)
    x0 = Conv2D(1, kernel_size=1, padding="same", name="depth_map_t0")(x0)

    x1 = Conv2D(2, (3, 3), padding="same", name="depth_map_t1_conv2d", use_bias=False)(x1)
    x1 = BatchNormalization(name="depth_map_t1_batchnorm")(x1)
    x1 = ReLU()(x1)
    x1 = Conv2D(1, kernel_size=1, padding="same", name="depth_map_t1")(x1)

    # Motion Network
    # =======================================
    x = Concatenate()([input_t0, x0, input_t1, x1])
    x, mn_fms = encoder(2, x, namescope="mm")

    mm = Conv2D(2, (3, 3), padding="same", name="mm_conv2d", use_bias=False)(x)
    mm = BatchNormalization(name="mm_batchnorm")(mm)
    mm = ReLU()(mm)
    mm = Conv2D(3, kernel_size=1, padding="same", name="motion_map")(mm)

    rot = Conv2D(2, (3, 3), name="rot_conv2d", use_bias=False)(mn_fms[-1])
    rot = BatchNormalization(name="rot_batchnorm")(rot)
    rot = ReLU()(rot)
    rot = Flatten()(rot)
    rot = Dense(128)(rot)
    rot = Dense(32)(rot)
    rot = Dense(3)(rot)

    tran = Conv2D(2, (3, 3), name="tran_conv2d", use_bias=False)(mn_fms[-1])
    tran = BatchNormalization(name="tran_batchnorm")(tran)
    tran = ReLU()(tran)
    tran = Flatten()(tran)
    tran = Dense(128)(tran)
    tran = Dense(32)(tran)
    tran = Dense(3)(tran)

    model = DmdsModel(inputs=[input_t0, input_t1, intr], outputs=[x0, x1, mm, rot, tran])

    return model


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))
