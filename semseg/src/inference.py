import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from pymongo import MongoClient
from semseg.src.processor import ProcessImages
from data.semseg_spec import SEMSEG_CLASS_MAPPING

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)

MODEL_PATH = "/home/jodo/trained_models/semseg_27-09-2020-08-38-47/tf_model_5"
IMG_WIDTH = 320
IMG_HEIGHT = 130
OFFSET_TOP = 60 # from original size (640x380)


def to_3channel(raw_semseg_img):
  array = np.zeros((raw_semseg_img.shape[0] * raw_semseg_img.shape[1] * 3), dtype='uint8')
  flattened_arr = raw_semseg_img.reshape((-1, len(SEMSEG_CLASS_MAPPING)))
  for i, one_hot_encoded_arr in enumerate(flattened_arr):
    # find index of highest value in the one_hot_encoded_arr
    cls_idx = np.argmax(one_hot_encoded_arr)
    # convert index to hex value
    hex_val = int(list(SEMSEG_CLASS_MAPPING.items())[int(round(cls_idx))][1])
    # fill new array with BGR values
    new_i = i * 3
    array[new_i] = (hex_val & 0xFF0000) >> 16
    array[new_i + 1] = (hex_val & 0x00FF00) >> 8
    array[new_i + 2] = (hex_val & 0x0000FF)

  return array.reshape((raw_semseg_img.shape[0], raw_semseg_img.shape[1], 3))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Inference from tensorflow model")
  parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
  parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
  parser.add_argument("--collection", type=str, default="kitti", help="MongoDB collection")
  args = parser.parse_args()

  client = MongoClient(args.conn)
  collection = client[args.db][args.collection]

  model: tf.keras.models.Model = tf.keras.models.load_model(MODEL_PATH, compile=False)
  model.summary()

  documents = collection.find({}).limit(20)
  for doc in documents:
    decoded_img = np.frombuffer(doc["img"], np.uint8)
    img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
    img, roi = ProcessImages.resize_img(img, IMG_WIDTH, IMG_HEIGHT, OFFSET_TOP)
    img = np.array([img])

    raw_result = model.predict(img)
    semseg_img = raw_result[0]
    semseg_img = to_3channel(semseg_img)  # convert hex to 3 channel representation

    cv2.imshow("Input Image", img[0])
    cv2.imshow("Semseg Image", semseg_img)
    cv2.waitKey(0)
