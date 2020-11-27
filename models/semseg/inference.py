import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from pymongo import MongoClient
from common.utils import to_3channel, resize_img
from data.semseg_spec import SEMSEG_CLASS_MAPPING

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="semseg", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="comma10k", help="MongoDB collection")
    parser.add_argument("--img_width", type=int, default=320, help="Width of image, must be model input")
    parser.add_argument("--img_height", type=int, default=130, help="Width of image, must be model input")
    parser.add_argument("--offset_bottom", type=int, default=-200, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, help="Path to a tensorflow model folder")
    args = parser.parse_args()

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
    model.summary()

    documents = collection.find({}).limit(20)
    for doc in documents:
        decoded_img = np.frombuffer(doc["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
        img, roi = resize_img(img, args.img_width, args.img_height, args.offset_bottom)
        img = np.array([img])

        raw_result = model.predict(img)
        semseg_img = raw_result[0]
        semseg_img = to_3channel(semseg_img, SEMSEG_CLASS_MAPPING)

        cv2.imshow("Input Image", img[0])
        cv2.imshow("Semseg Image", semseg_img)
        cv2.waitKey(0)
