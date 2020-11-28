import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from pymongo import MongoClient
from common.utils import to_3channel, resize_img
from data.od_spec import OD_CLASS_MAPPING
from models.centernet_2d import process_2d_output

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4864)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="kitti", help="MongoDB collection")
    parser.add_argument("--model_path", type=str, help="Path to a tensorflow model folder")
    args = parser.parse_args()

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
    model.summary()

    input_shape = model.input.shape
    output_shape = model.output.shape
    nb_classes = len(OD_CLASS_MAPPING)

    documents = collection.find({}).limit(20)
    for doc in documents:
        decoded_img = np.frombuffer(doc["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
        input_img, roi = resize_img(img, input_shape[2], input_shape[1])

        raw_result = model.predict(np.array([input_img]))
        output_mask = raw_result[0]
        heatmap = to_3channel(output_mask, OD_CLASS_MAPPING)
        objects = process_2d_output(output_mask, roi, 2.0, nb_classes)

        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["obj_idx"]]
            cv2.rectangle(img, obj["top_left"], obj["bottom_right"], (color[2], color[1], color[0]), 1)

        cv2.imshow("Input Image with Objects", img)
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(0)