import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.engine import data_adapter
from models.dmds import DmdsParams
from models.dmds.convert import create_dataset
from common.utils import tflite_convert, resize_img
from common.layers import encoder, upsample_block, bottle_neck_block


# Creating custom train_step() in order to call loss function with all needed parameters
class DmdsModel(Model):
    def init_file_writer(self, logdir):
        self.file_writer = tf.summary.create_file_writer(logdir)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        in_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(in_data, training=True)
            loss = self.custom_loss.calc(in_data[0], in_data[1], y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], y_pred[5], gt[0], gt[1])

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            img0 = in_data[0] / 255.0
            c0 = tf.unstack(img0, axis=-1)
            img0 = tf.stack([c0[2], c0[1], c0[0]], axis=-1)
            img1 = in_data[1] / 255.0
            c1 = tf.unstack(img1, axis=-1)
            img1 = tf.stack([c1[2], c1[1], c1[0]], axis=-1)
            depth0 = y_pred[0] / 100.0
            depth1 = y_pred[1] / 100.0
            tf.summary.image("img0", img0, max_outputs=4, step=0)
            tf.summary.image("img1", img1, max_outputs=4, step=0)
            tf.summary.image("depth0", depth0, max_outputs=4, step=0)
            tf.summary.image("depth1", depth1, max_outputs=4, step=0)
            tf.summary.histogram("depth_hist", depth0, step=0)


        loss_dict = self.custom_loss.loss_vals
        loss_dict["sum"] = loss
        return loss_dict
    
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        in_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(in_data, training=False)
        loss = self.custom_loss.calc(in_data[0], in_data[1], y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], y_pred[5], gt[0], gt[1])

        loss_dict = self.custom_loss.loss_vals
        loss_dict["sum"] = loss
        return loss_dict


def create_model(input_height: int, input_width: int, base_model_path: str = None) -> tf.keras.Model:
    # Depth Maps
    # =======================================
    intr = Input(shape=(3, 3))
    input_t0 = Input(shape=(input_height, input_width, 3))
    input_t1 = Input(shape=(input_height, input_width, 3))
    x0, _ = encoder(8, input_t0, namescope="depth_t0")
    x1, _ = encoder(8, input_t1, namescope="depth_t1")
    # depth_model_t0 = Model(inputs=input_t0, outputs=x0)
    # depth_model_t1 = Model(inputs=input_t1, outputs=x1)

    x0 = Conv2D(32, (3, 3), padding="same", name="depth_map_t0_conv2d")(x0)
    x0 = ReLU()(x0)
    x0 = BatchNormalization()(x0)
    x0 = Conv2D(1, kernel_size=1, padding="same", activation=None, name="depth_map_t0")(x0)

    x1 = Conv2D(32, (3, 3), padding="same", name="depth_map_t1_conv2d")(x1)
    x1 = ReLU()(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(1, kernel_size=1, padding="same", activation=None, name="depth_map_t1")(x1)

    # Motion Network
    # =======================================
    x = Concatenate()([input_t0, x0, input_t1, x1])
    x, mn_fms = encoder(8, x, namescope="mm")

    mm = Conv2D(16, (3, 3), padding="same", name="mm_conv2d", use_bias=False)(x)
    mm = BatchNormalization(name="mm_batchnorm")(mm)
    mm = ReLU()(mm)
    mm = Conv2D(3, kernel_size=1, padding="same", name="motion_map")(mm)

    rot = Conv2D(64, (3, 3), strides=(2, 2), name="rot1_conv2d", use_bias=False)(mn_fms[-1])
    rot = BatchNormalization()(rot)
    rot = ReLU()(rot)
    rot = Conv2D(64, (3, 3), strides=(2, 2), name="rot2_conv2d", use_bias=False)(rot)
    rot = BatchNormalization()(rot)
    rot = ReLU()(rot)
    rot = Flatten()(rot)
    rot = Dropout(0.35)(rot)
    rot = Dense(64, activation="relu")(rot)
    rot = Dropout(0.3)(rot)
    rot = Dense(16, activation="relu")(rot)
    rot = Dropout(0.15)(rot)
    rot = Dense(3)(rot)

    tran = Conv2D(64, (3, 3), strides=(2, 2), name="tran1_conv2d", use_bias=False)(mn_fms[-1])
    tran = BatchNormalization()(tran)
    tran = ReLU()(tran)
    tran = Conv2D(64, (3, 3), strides=(2, 2), name="tran2_conv2d", use_bias=False)(tran)
    tran = BatchNormalization()(tran)
    tran = ReLU()(tran)
    tran = Flatten()(tran)
    tran = Dropout(0.35)(tran)
    tran = Dense(64, activation="relu")(tran)
    tran = Dropout(0.3)(tran)
    tran = Dense(16, activation="relu")(tran)
    tran = Dropout(0.15)(tran)
    tran = Dense(3)(tran)

    model = DmdsModel(inputs=[input_t0, input_t1, intr], outputs=[x0, x1, mm, rot, tran, intr])

    return model


if __name__ == "__main__":
    params = DmdsParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.summary()
    plot_model(model, to_file="./tmp/dmds_model.png")
    tflite_convert(model, "./tmp", True, True, create_dataset(model.input[0].shape))
