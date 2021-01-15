import tensorflow as tf
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import argparse
import time
from pymongo import MongoClient
from common.utils import to_3channel, resize_img
from data.od_spec import OD_CLASS_MAPPING
from models.centernet_2d import process_2d_output, Params

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="kitti_test", help="MongoDB collection")
    parser.add_argument("--offset_bottom", type=int, default=-130, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/path/to/tf_model_x", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    is_tf_lite = args.model_path[-7:] == ".tflite"
    model = None
    interpreter = None
    input_details = None
    output_details = None
    input_shape = None
    output_shape = None
    nb_classes = len(OD_CLASS_MAPPING)

    if is_tf_lite:
        # Load the TFLite model and allocate tensors.
        if args.use_edge_tpu:
            print("Using EdgeTpu")
            interpreter = edgetpu.make_interpreter(args.model_path)
        else:
            print("Using TFLite")
            interpreter = tflite.Interpreter(args.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']
    else:
        model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
        model.summary()
        input_shape = model.input.shape
        output_shape = model.output.shape
        print("Using Tensorflow")

    # alternative data source, mp4 video
    # cap = cv2.VideoCapture('/path/to/example.mp4')
    # while (cap.isOpened()):
    #     ret, img = cap.read()
    documents = collection.find({}).limit(20)
    for doc in documents:
        decoded_img = np.frombuffer(doc["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)

        input_img, roi = resize_img(img, input_shape[2], input_shape[1], offset_bottom=args.offset_bottom)

        if is_tf_lite:
            input_img = input_img.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], [input_img])
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_mask = output_data[0]
        else:
            start_time = time.time()
            raw_result = model.predict(np.array([input_img]))
            elapsed_time = time.time() - start_time
            output_mask = raw_result[0]

        heatmap = to_3channel(output_mask, OD_CLASS_MAPPING)
        r = float(input_shape[1]) / float(output_shape[1])
        objects = process_2d_output(output_mask, roi, r, nb_classes)
        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["obj_idx"]]
            cv2.rectangle(img, obj["top_left"], obj["bottom_right"], (color[2], color[1], color[0]), 1)

        print(str(elapsed_time) + " s")
        cv2.imshow("Org Image with Objects", img)
        cv2.imshow("Input Image", input_img)
        cv2.imshow("Heatmap", heatmap)
        cv2.waitKey(0)
