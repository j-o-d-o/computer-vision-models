import os
import tensorflow as tf
from tensorflow.keras import optimizers, models
from mlpipe.data_reader.mongodb import load_ids, MongoDBGenerator
from mlpipe.utils import MLPipeLogger, Config
from ssd.src.processor import ProcessImage, GenGroundTruth
from ssd.src.params import Params
from ssd.src.model import create_model
from ssd.src.prior_boxes import PriorBoxes
from ssd.src.loss import SSDLoss
from ssd.src.save_callback import SaveToStorage

print("Using Tensorflow Version: " + tf.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":
  MLPipeLogger.init()
  MLPipeLogger.remove_file_logger()

  Config.add_config('./../../config.ini')
  collection_details = ("localhost_mongo_db", "object_detection", "kitty_training")

  # Create Data Generators
  train_data, val_data = load_ids(
    collection_details,
    data_split=(80, 20),
    # limit=100,
  )

  prior_boxes = PriorBoxes(clip_boxes=True)
  processors = [ProcessImage(), GenGroundTruth(prior_boxes)]
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
  loss = SSDLoss(
    num_boxes=prior_boxes.get_num_boxes(),
    num_classes=len(Params.CLASSES),
    alpha=1,
    num_neg_min=10,
  ).compute_loss

  opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=5e-06)
  custom_objects = {"compute_loss": loss}

  if Params.LOAD_PATH is None:
    model: models.Model = create_model()
    model.compile(optimizer=opt, loss=loss)
    model.summary()
  else:
    # Load model (currently only Keras h5 format works due to Tensorflow issues)
    # Issue: https://github.com/tensorflow/tensorflow/pull/34048
    model: models.Model = models.load_model(Params.LOAD_PATH, custom_objects=custom_objects)
    model.summary()

  # Train Model
  callbacks = [(SaveToStorage("/home/jodo/trained_models", "kitti_mobile_ssd", model, prior_boxes, False, custom_objects=custom_objects))]

  model.fit_generator(
    generator=train_gen,
    validation_data=val_gen,
    epochs=Params.NUM_EPOCH,
    verbose=1,
    callbacks=callbacks,
    initial_epoch=0,
    use_multiprocessing=True,
    workers=2,
  )
