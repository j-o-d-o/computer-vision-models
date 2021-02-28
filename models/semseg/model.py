import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Concatenate, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from data.label_spec import SEMSEG_CLASS_MAPPING
from common.layers import bottle_neck_block, upsample_block
from numba.typed import List
from models.semseg import SemsegParams
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from common.utils import to_3channel
from tensorflow.python.keras.engine import compile_utils


class SemsegModel(Model):
    def init_save_dir(self, save_dir):
        self.save_dir = save_dir
        self.file_writer = tf.summary.create_file_writer(save_dir)
        self.train_step_counter = 0

    def update_custom_metrics(self, y_true, y_pred, pos_mask):
        ce_value = tf.reduce_sum(tf.keras.metrics.categorical_crossentropy(y_true, y_pred) * pos_mask) / tf.reduce_sum(pos_mask)
        self.custom_metrics[0].update_state(ce_value)

    def compile(self, optimizer, custom_loss):
        super().compile(optimizer)
        self.custom_loss = custom_loss
        self.custom_metrics = [tf.keras.metrics.Mean("ce")]

    @property
    def metrics(self):
        return_val = list(self.custom_loss.metrics.values())
        return_val += self.custom_metrics
        return return_val

    def train_step(self, data):
        self.train_step_counter += 1
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_mask = gt[0]
        pos_mask = gt[1]

        with backprop.GradientTape() as tape:
            semseg_pred = self(input_data, training=True)
            loss_val = self.custom_loss.calc(input_data, semseg_mask, pos_mask, semseg_pred)
        self.optimizer.minimize(loss_val, self.trainable_variables, tape=tape)

        # Using the file writer, log images
        if self.train_step_counter % 200 == 0:
            tf.summary.experimental.set_step(self.train_step_counter)
            with self.file_writer.as_default():
                inp_img = cv2.cvtColor(input_data[0].numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
                semseg_true_img = cv2.cvtColor(to_3channel(semseg_mask[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB)
                semseg_pred_img = cv2.cvtColor(to_3channel(semseg_pred[0].numpy(), List(SEMSEG_CLASS_MAPPING.items())), cv2.COLOR_BGR2RGB)

                tf.summary.image("inp", np.expand_dims(inp_img, axis=0), max_outputs=80)
                tf.summary.image("true", np.expand_dims(semseg_true_img, axis=0), max_outputs=80)
                tf.summary.image("pred", np.expand_dims(semseg_pred_img, axis=0), max_outputs=80)

        self.update_custom_metrics(semseg_mask, semseg_pred, pos_mask)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # self.train_step_counter = 0
        data = data_adapter.expand_1d(data)
        input_data, gt, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        semseg_mask = gt[0]
        pos_mask = gt[1]

        semseg_pred = self(input_data, training=False)
        loss_val = self.custom_loss.calc(input_data, semseg_mask, pos_mask, semseg_pred)

        self.update_custom_metrics(semseg_mask, semseg_pred, pos_mask)
        return {m.name: m.result() for m in self.metrics}


def create_model(input_height: int, input_width: int) -> tf.keras.Model:
    """
    Create a semseg model
    :param input_height: Height of the input image
    :param input_width: Width of the input image
    :return: Semseg Keras Model
    """
    inp = Input(shape=(input_height, input_width, 3))
    inp_rescaled = tf.keras.layers.experimental.preprocessing.Rescaling(scale=255.0, offset=0)(inp)
    fms = [] # [inp_rescaled]
    namescope = "semseg/"
    filters = 8

    # Downsample
    # ----------------------------
    x = Conv2D(filters, 5, padding="same", name=f"{namescope}initial_downsample", strides=(2, 2), kernel_regularizer=l2(l=0.0001))(inp_rescaled)
    x = BatchNormalization(name=f"{namescope}initial_batchnorm")(x)
    x = ReLU(6., name=f"{namescope}initial_acitvation")(x)
    fms.append(x)

    x = bottle_neck_block(f"{namescope}downsample_0/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_1/", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"{namescope}downsample_2/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_3/", x, filters, downsample = True)
    fms.append(x)
    filters = int(filters * 2)

    x = bottle_neck_block(f"{namescope}downsample_4/", x, filters)
    x = bottle_neck_block(f"{namescope}downsample_5/", x, filters, downsample = True)
    filters = int(filters * 2)

    # Dilation (to avoid further downsampling)
    # ----------------------------
    dilation_rates = [3, 6, 9, 12]
    concat_tensors = []
    x = Conv2D(filters, kernel_size=1, name=f"{namescope}start_dilation_1x1", kernel_regularizer=l2(l=0.0001))(x)
    concat_tensors.append(x)
    for i, rate in enumerate(dilation_rates):
        x = bottle_neck_block(f"{namescope}dilation_{i}", x, filters)
        x = BatchNormalization(name=f"{namescope}dilation_batchnorm_{i}")(x)
        concat_tensors.append(x)

    x = Concatenate(name=f"{namescope}dilation_concat")(concat_tensors)
    fms.append(x)

    # Upsample
    # ----------------------------
    for i in range(len(fms) - 2, -1, -1):
        fms[i] = Conv2D(filters, (3, 3), padding="same", name=f"{namescope}conv2d_up_{i}", kernel_regularizer=l2(l=0.0001))(fms[i])
        fms[i] = BatchNormalization(name=f"{namescope}batchnorm_up_{i}")(fms[i])
        fms[i] = ReLU(6.0)(fms[i])
        x = upsample_block(f"{namescope}upsample_{i}/", x, fms[i], filters)
        filters = int(filters // 2)

    # Create Semseg Map
    # ----------------------------
    semseg_map = Conv2D(len(SEMSEG_CLASS_MAPPING), kernel_size=1, name="semseg/out", activation="sigmoid", kernel_regularizer=l2(l=0.0001))(x)

    return SemsegModel(inputs=[inp], outputs=semseg_map)


# To test model creation and quickly check if edgetpu compiler compiles it correctly
if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    from models.semseg import convert
    from common.utils import set_weights, tflite_convert
    
    params = SemsegParams()
    model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    set_weights.set_weights("/home/computer-vision-models/trained_models/semseg_comma10k_augment_2021-02-28-10235/tf_model_54/keras.h5", model, force_resize=True, custom_objects={"SemsegModel": SemsegModel})
    model.summary()
    plot_model(model, to_file="./tmp/semseg_model.png")
    tflite_convert.tflite_convert(model, "./tmp", True, True, convert.create_dataset(model.input.shape))
