import tensorflow as tf
from tensorflow.keras import optimizers, models, metrics
from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.callbacks import SaveToStorage
from common.utils import Logger, Config
from models.semseg import create_model, Params, ProcessImages, SemsegLoss

print("Using Tensorflow Version: " + tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "semseg", "comma10k")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(77, 23),
        shuffle_data=True,
    )

    processors = [ProcessImages()]
    train_gen = MongoDBGenerator(
        collection_details,
        train_data,
        batch_size=Params.BATCH_SIZE,
        processors=processors
    )
    val_gen = MongoDBGenerator(
        collection_details,
        val_data,
        batch_size=Params.BATCH_SIZE,
        processors=processors
    )

    # Create Model
    loss = SemsegLoss()
    metrics = []
    opt = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    if Params.LOAD_PATH is None:
        model: models.Model = create_model()
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        model.summary()
    else:
        custom_objects = {"compute_loss": loss}
        model: models.Model = models.load_model(Params.LOAD_PATH, custom_objects=custom_objects)
        model.summary()

    # for debugging custom loss or layers, set to True
    # model.run_eagerly = True

    # Train model
    name = ""
    storage_path = "./trained_models/semseg_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, False), tensorboard_callback]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Params.NUM_EPOCH,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        use_multiprocessing=False,
        workers=2,
    )
