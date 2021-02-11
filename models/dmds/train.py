import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config, set_up_tf_gpu
from models.dmds import create_model, DmdsParams, ProcessImages, DmdsLoss


def adapt_doc_ids(doc_ids):
    adapted_doc_ids = []
    for i in range(0, len(doc_ids) - 1):
        t0 = doc_ids[i]
        t1 = doc_ids[i+1]
        adapted_doc_ids.append(t0)
        adapted_doc_ids.append(t1)
        adapted_doc_ids.append(t1)
        adapted_doc_ids.append(t0)
    return adapted_doc_ids

if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = DmdsParams()

    # get one entry from the database
    Config.add_config('./config.ini')
    con = ("local_mongodb", "depth", "driving_stereo")
    scenes = [
        "2018-10-19-09-30-39",
        "2018-10-22-10-44-02"
    ]
    train_data = []
    val_data = []
    collection_details = []

    # get ids
    for scene_token in scenes:
        td, vd = load_ids(
            con,
            data_split=(70, 30),
            shuffle_data=False,
            mongodb_filter={"scene_token": scene_token},
            sort_by={"timestamp": 1},
            limit=100
        )
        train_data = adapt_doc_ids(train_data)
        val_data = adapt_doc_ids(val_data)
        train_data.append(td)
        val_data.append(vd)
        collection_details.append(con)

    processors = [ProcessImages(params)]
    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=params.BATCH_SIZE,
        processors=processors,
        data_group_size=2,
        shuffle_data=False
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=params.BATCH_SIZE,
        processors=processors,
        data_group_size=2,
        shuffle_data=False
    )

    # Create Model
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    if params.LOAD_PATH is not None:
        with tfmot.quantization.keras.quantize_scope():
            custom_objects = {"SemsegLoss": loss}
            model: models.Model = models.load_model(params.LOAD_PATH, custom_objects=custom_objects, compile=False)
    else:
        model: models.Model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)

    model.compile(optimizer=opt, metrics=[])
    model.summary()

    # model.run_eagerly = True

    # Train model
    storage_path = "./trained_models/dmds_ds_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, True), tensorboard_callback]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        workers=1,
        # use_multiprocessing=True
    )
