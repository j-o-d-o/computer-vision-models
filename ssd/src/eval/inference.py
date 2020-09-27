import tensorflow as tf
from tensorflow.keras import models, losses
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.keras.saving.saved_model import load as saved_model_load
import numpy as np
import configparser
import cv2
from pymongo import MongoClient
from typing import List
from ssd.src.prior_boxes import PriorBoxDecoder
from ssd.src.params import Params
from ssd.src.processor import ProcessImage

color_code = [
  (0, 255, 0),
  (255, 0, 0),
  (0, 0, 255),
  (255, 255, 0),
  (0, 255, 255),
  (255, 0, 255),
]

BASE_PATH = "/home/jodo/trained_models/kitti_mobile_ssd_31-01-2020-06-58-03"
H5_PATH = BASE_PATH + "/keras_model_21.h5"
PRIOR_BOX_PATH = BASE_PATH + "/prior_boxes.json"

tf.enable_eager_execution() # not needed anymore in TF 2.0
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def show_img(test_img, detected_boxes: List[List[float]] = None):
  test_img = test_img.astype(np.uint8)
  if detected_boxes is not None:
    for box in detected_boxes:
      cx, cy, w, h = box[0] * Params.IMG_WIDTH
      score = np.max(box[1][1:])
      class_idx = np.argmax(box[1][1:])

      overlay = test_img.copy()
      cv2.rectangle(overlay, (int(cx - w / 2), int(cy + h / 2)), (int(cx + w / 2), int(cy - h / 2)), color_code[class_idx], 1)
      test_img = cv2.addWeighted(overlay, score, test_img, 1 - score, 0)

  cv2.imshow("Test Img", test_img)
  cv2.waitKey(0)


if __name__ == "__main__":
  # Read config file
  config_path = "../../config.ini"
  cp = configparser.ConfigParser()
  if not len(cp.read(config_path)) == 0:
    config_name = "test_data"
    CLIENT = MongoClient(cp[config_name]["connection"])
    COLLECTION = CLIENT[cp[config_name]["database"]][cp[config_name]["collection"]]
  else:
    raise ValueError("Config file not found for path: " + config_path)

  # Create Prior Boxes
  decoder = PriorBoxDecoder()
  decoder.init_boxes_from_file(PRIOR_BOX_PATH)

  # Load model (not we have to put in a dummy loss function for the custom one, but it is not used during inference)
  model: tf.keras.models.Model = tf.keras.models.load_model(H5_PATH, compile=False)
  model.summary()

  documents = COLLECTION.find({}).limit(20)
  for doc in documents:
    decoded_img = np.frombuffer(doc["img"], np.uint8)
    img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
    img, roi = ProcessImage.resize_img(img)
    img = np.array([img]).astype(np.float32)

    raw_result = model.predict(img)
    boxes = decoder.decode_boxes(raw_result[0], len(Params.CLASSES), threshold=0.2)

    show_img(img[0], boxes)
    cv2.waitKey(0)
