import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.processors import AugmentImages
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config, set_up_tf_gpu
from models.semseg import create_model, SemsegParams, ProcessImages, SemsegLoss


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = SemsegParams()

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "semseg", "comma10k")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(88, 12),
        shuffle_data=True,
    )

    processors = [ProcessImages(params)]
    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=processors
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=processors
    )

    # Create Model
    loss = SemsegLoss()
    opt = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    if params.LOAD_PATH is not None:
        with tfmot.quantization.keras.quantize_scope():
            custom_objects = {"SemsegLoss": loss}
            model: models.Model = models.load_model(params.LOAD_MODEL_PATH, custom_objects=custom_objects, compile=False)
    else:
        model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)

    model.compile(optimizer=opt, loss=loss, metrics=[])
    model.summary()

    # for debugging custom loss or layers, set to True
    # model.run_eagerly = True

    # Train model
    storage_path = "./trained_models/semseg_comma10k_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=2,
    )
