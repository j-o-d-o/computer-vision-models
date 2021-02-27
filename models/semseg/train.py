import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config
from models.semseg import create_model, SemsegParams, ProcessImages, SemsegLoss


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = SemsegParams()

    Config.add_config('./config.ini')
    collection_details = ("aws_mongodb", "labels", "comma10k")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(85, 15),
        shuffle_data=True
    )

    train_gen = MongoDBGenerator(
        [collection_details],
        [train_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, 9)],
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
    loss = SemsegLoss(save_path=storage_path)

    if params.LOAD_PATH is not None:
        with tfmot.quantization.keras.quantize_scope():
            custom_objects = {"SemsegLoss": loss}
            model: models.Model = models.load_model(params.LOAD_PATH, custom_objects=custom_objects, compile=False)
    else:
        model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, params.LOAD_WEIGHTS)

    model.compile(optimizer=opt, custom_loss=loss)
    model.summary()
    # for debugging custom loss or layers, set to True
    model.run_eagerly = True
    model.init_save_dir(storage_path + "/images")

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
