import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config
from models.depth import create_model, Params, ProcessImages
from models.depth.loss import DepthLoss


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = Params()

    # get one entry from the database
    Config.add_config('./config.ini')
    # con = ("local_mongodb", "labels", "driving_stereo")
    con = ("local_mongodb", "labels", "nuscenes_train")

    td, vd = load_ids(
        con,
        data_split=(90, 10),
        shuffle_data=True
    )
    train_data = [td]
    val_data = [vd]
    collection_details = [con]

    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)],
        shuffle_data=True
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params, False)],
        shuffle_data=True
    )

    # Create Model
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    if params.LOAD_PATH is not None:
        model: models.Model = models.load_model(params.LOAD_PATH, compile=False)
    else:
        model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH, params.LOAD_PATH_WEIGHTS)

    # Train model
    storage_path = "./trained_models/depth_ds_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]

    model.compile(optimizer=opt, loss=DepthLoss(save_path=storage_path))
    model.summary()

    model.run_eagerly = True

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
