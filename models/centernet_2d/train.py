import tensorflow as tf
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Logger, Config
from common.callbacks import SaveToStorage
from data.od_spec import OD_CLASS_MAPPING
from models.centernet_2d import ProcessImages, Params, Centernet2DLoss, create_model

print("Using Tensorflow Version: " + tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4864)])


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "object_detection", "kitti")

    # Create Data Generators
    train_data, val_data = load_ids(
        collection_details,
        data_split=(77, 23),
        shuffle_data=True
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

    nb_classes = len(OD_CLASS_MAPPING)
    loss = Centernet2DLoss(nb_classes, Params.LOSS_SIZE_WEIGHT, Params.FOCAL_LOSS_ALPHA, Params.FOCAL_LOSS_BETA)
    opt = tf.keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-07) 

    if Params.LOAD_PATH is None:
        model: tf.keras.models.Model = create_model(
            Params.INPUT_HEIGHT,
            Params.INPUT_WIDTH,
            int(Params.INPUT_HEIGHT // Params.R),
            int(Params.INPUT_WIDTH // Params.R),
            nb_classes
        )
        model.compile(optimizer=opt, loss=loss)
        # model.summary()
    else:
        custom_objects = {"compute_loss": loss}
        model: tf.keras.models.Model = models.load_model(Params.LOAD_PATH, custom_objects=custom_objects)
        # model.summary()

    # Train Model
    callbacks = [(SaveToStorage("./trained_models", "centernet2d", model, False))]
    model.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=Params.NUM_EPOCH,
        verbose=1,
        callbacks=[callbacks],
        initial_epoch=0,
        use_multiprocessing=False,
        workers=2,
    )