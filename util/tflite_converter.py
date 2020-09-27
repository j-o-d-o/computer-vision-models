import tensorflow as tf
import numpy as np
import cv2
from pymongo import MongoClient
from semseg.params import Params
from semseg.processor import ProcessImages

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)


def convert(source_path, dataset):
  converter = tf.lite.TFLiteConverter.from_saved_model(source_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  def representative_dataset_gen():
    # Get sample input data as a numpy array in a method of your choosing.
    yield [dataset[0]]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8

  tflite_model = converter.convert()
  open(arg_source_path + "/model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
  client = MongoClient("mongodb://localhost:27017")
  collection = client["object_detection"]["comma10k"]
  documents = collection.find({}).limit(200)
  arg_dataset = []
  for doc in documents:
    decoded_img = np.frombuffer(doc["img"], np.uint8)
    img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
    img, roi = ProcessImages.resize_img(img, Params.IMG_WIDTH, Params.IMG_HEIGHT)
    img = img.astype(np.float32)
    arg_dataset.append([img])

  arg_source_path = "/home/jodo/trained_models/semseg_16-08-2020-13-16-55/tf_model_36"
  convert(arg_source_path, arg_dataset)

  # The TFLite model that is generated here does not yet run on the edge tpu, it needs one more compile step as explained here
  # https://coral.ai/docs/edgetpu/compiler
  # The TLDR;
  #  >> curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  #  >> echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
  #  >> sudo apt-get update
  #  >> sudo apt-get install edgetpu-compiler
  #  >> edgetpu-compiler path/to/model_quant.tflite
  # TODO: Add this compile step here
