# Note: Make sure to follow the instructions on the README.md for edgetpu support
import tensorflow as tf
import numpy as np
import cv2
from pymongo import MongoClient
from common.utils import resize_img
from models.semseg.params import Params

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)


def convert(source_path, dataset):
  converter = tf.lite.TFLiteConverter.from_saved_model(source_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  def representative_dataset_gen():
    # Get sample input data as a numpy array in a method of your choosing.
    for data in dataset:
      yield [data]
  converter.representative_dataset = representative_dataset_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8

  tflite_model = converter.convert()
  open(source_path + "/model_quant_int8.tflite", "wb").write(tflite_model)

def main(con_str, db_str, collection_str, input_width, input_height):
  client = MongoClient(con_str)
  collection = client[db_str][collection_str]
  documents = collection.find({}).limit(200)

  dataset = []
  documents_list = list(documents)
  assert(len(documents_list) > 0)
  for doc in documents_list:
    decoded_img = np.frombuffer(doc["img"], np.uint8)
    img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
    img, roi = resize_img(img, input_width, input_height)
    img = img.astype(np.float32)
    dataset.append([img])

  arg_source_path = "/home/jo/git/computer-vision-models/trained_models/semseg_2020-12-04-081334/tf_model_22"
  convert(arg_source_path, dataset)

if __name__ == "__main__":
  main("mongodb://localhost:27017", "semseg", "comma10k", Params.INPUT_WIDTH, Params.INPUT_HEIGHT)
