import tensorflow as tf
import numpy as np
import cv2
import argparse
import subprocess
from pymongo import MongoClient
from common.utils import resize_img

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


def main(model_path: str, con_str: str, db_str: str, collection_str: str, quantize: bool, compile_edge_tpu_flag: bool):
  """
  Convert and quantize tensorflow model to tflite and compile for edgetpu
  :param model_path: path to the tensorflow model (containing the model.pb, variables and assets folder)
  :param con_str: mongodb connection string
  :param db_str: mongodb database string
  :param collection_str: mongodb collection string
  :param quantize: bool flag if the model should be quantized
  :param compile_edge_tpu_flag: bool flag if the tflite model should also be compiled for edge tpu
  """
  # Convert to TFLite
  converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  if quantize:
    # MongoDB connection for sample dataset
    client = MongoClient(con_str)
    collection = client[db_str][collection_str]
    documents = collection.find({}).limit(200)

    # Get the model input size for resizing images
    model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
    input_shape = model.input.shape
    print("Resize Input to: " + str(input_shape))

    # Create sample dataset for post training quantization
    dataset = []
    documents_list = list(documents)
    assert(len(documents_list) > 0)
    for doc in documents_list:
      decoded_img = np.frombuffer(doc["img"], np.uint8)
      img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
      img, roi = resize_img(img, input_shape[2], input_shape[1])
      img = img.astype(np.float32)
      dataset.append([img])

    def representative_dataset_gen():
      # Get sample input data as a numpy array in a method of your choosing.
      for data in dataset:
        yield [data]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    tflite_model_path = model_path + "/model_quant.tflite"
    open(tflite_model_path, "wb").write(tflite_model)

    # Compile for EdgeTpu
    if compile_edge_tpu_flag:
      print("Compile for EdgeTpu")
      subprocess.run("edgetpu_compiler -o %s %s" % (model_path, tflite_model_path), shell=True)
  else:
    tflite_model = converter.convert()
    tflite_model_path = model_path + "/model.tflite"
    open(tflite_model_path, "wb").write(tflite_model)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Convert and quantize tensorflow model to tflite and edgetpu")
  parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
  parser.add_argument("--db", type=str, default="semseg", help="MongoDB database")
  parser.add_argument("--collection", type=str, default="comma10k", help="MongoDB collection")
  parser.add_argument("--model_path", type=str, default="/home/jo/git/computer-vision-models/trained_models/semseg_2021-01-01-11203/tf_model_20", help="Path to a tensorflow model folder")
  parser.add_argument("--quantize", action="store_true", help="Quantize model using input data")
  parser.add_argument("--compile_edge_tpu", action="store_true", help="Compile TFLite model also for EdgeTpu")
  args = parser.parse_args()

  # args.compile_edge_tpu = False
  # args.quantize = False

  main(args.model_path, args.conn, args.db, args.collection, args.quantize, args.compile_edge_tpu)
