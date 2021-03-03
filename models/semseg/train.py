import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

import tensorflow_model_optimization as tfmot
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config, set_weights
from models.semseg import create_model, SemsegParams, SemsegLoss, ProcessImages, ShowPygame


def make_custom_callbacks(keras_model, show_pygame):
    original_train_step = keras_model.train_step
    def call_custom_callbacks(original_data):
        data = data_adapter.expand_1d(original_data)
        x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        y_true_mask = y_true[:, :, :, :-1]
        y_pred = keras_model(x, training=True)
        result = original_train_step(original_data)
        # custom stuff called during training
        show_pygame.show_semseg(x, y_true_mask, y_pred)
        return result
    return call_custom_callbacks

if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = SemsegParams()

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "labels", "comma10k")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(90, 10),
        shuffle_data=True
    )

    train_gen = MongoDBGenerator(
        [collection_details],
        [train_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, [30, 60])],
        shuffle_data=True
    )
    val_gen = MongoDBGenerator(
        [collection_details],
        [val_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)],
        shuffle_data=True
    )

    # Create Model
    storage_path = "./trained_models/semseg_comma10k_augment_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    loss = SemsegLoss(params.CLASS_WEIGHTS)

    model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
    model.train_step = make_custom_callbacks(model, ShowPygame(storage_path + "/images"))
    model.compile(optimizer=opt, loss=loss)

    if params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.LOAD_WEIGHTS, model, force_resize=False)

    model.summary()
    # for debugging custom loss or layers, set to True
    model.run_eagerly = True

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=3,
        # use_multiprocessing=True
    )
