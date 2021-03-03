import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.config.optimizer.set_jit(True)

from datetime import datetime
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Logger, Config, set_weights
from common.callbacks import SaveToStorage
from data.label_spec import OD_CLASS_MAPPING
from models.centernet import ProcessImages, CenternetParams, CenternetLoss, create_model


if __name__ == "__main__":
    Logger.init()
    Logger.remove_file_logger()

    params = CenternetParams(len(OD_CLASS_MAPPING))
    params.REGRESSION_FIELDS["l_shape"].active = False
    params.REGRESSION_FIELDS["3d_info"].active = False

    Config.add_config('./config.ini')
    collection_details = ("local_mongodb", "labels", "nuimages")

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
        processors=[ProcessImages(params, 10)]
    )
    val_gen = MongoDBGenerator(
        [collection_details],
        [val_data],
        batch_size=params.BATCH_SIZE,
        processors=[ProcessImages(params)]
    )

    # Create Model
    storage_path = storage_path = "./trained_models/centernet_nuimages_" + datetime.now().strftime("%Y-%m-%d-%H%-M%-S")
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07) 
    loss = CenternetLoss(params)

    model: tf.keras.models.Model = create_model(params)
    model.compile(optimizer=opt, loss=loss, metrics=[loss.class_focal_loss, loss.r_offset_loss, loss.fullbox_loss])

    if params.LOAD_WEIGHTS is not None:
        set_weights.set_weights(params.LOAD_WEIGHTS, model)

    model.summary()
    model.run_eagerly = True

    # Train Model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=storage_path + "/tensorboard", histogram_freq=1)
    callbacks = [SaveToStorage(storage_path, model, False), tensorboard_callback]
    params.save_to_storage(storage_path)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=params.PLANED_EPOCHS,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        use_multiprocessing=False,
        workers=3,
    )
