import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter as tflite
from pycoral.utils import edgetpu
import numpy as np
import os
import cv2
import pygame
from pygame.locals import * 
import matplotlib.pyplot as plt
import argparse
import time
from pymongo import MongoClient
from common.utils import to_3channel, resize_img
from data.label_spec import SEMSEG_CLASS_MAPPING, OD_CLASS_MAPPING
from models.multitask import MultitaskParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference from tensorflow model")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="labels", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuscenes_train", help="MongoDB collection")
    parser.add_argument("--img_width", type=int, default=640, help="Width of image, must be model input")
    parser.add_argument("--img_height", type=int, default=256, help="Width of image, must be model input")
    parser.add_argument("--offset_bottom", type=int, default=0, help="Offset from the bottom in orignal image scale")
    parser.add_argument("--model_path", type=str, default="/path/to/tf_model_x/model_quant_edgetpu.tflite", help="Path to a tensorflow model folder")
    parser.add_argument("--use_edge_tpu", action="store_true", help="EdgeTpu should be used for inference")
    args = parser.parse_args()

    # For debugging force a value here
    args.use_edge_tpu = True
    args.model_path = "/home/computer-vision-models/tmp/model_quant_edgetpu.tflite"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]
    params = MultitaskParams(len(OD_CLASS_MAPPING.items()))

    is_tf_lite = args.model_path[-7:] == ".tflite"
    model = None
    interpreter = None
    input_details = None
    output_details = None

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
    else:
        with tfmot.quantization.keras.quantize_scope():
            model: tf.keras.models.Model = tf.keras.models.load_model(args.model_path, compile=False)
        model.summary()
        print("Using Tensorflow")

    # create pygame display to show images
    display = pygame.display.set_mode((args.img_width, args.img_height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)

    # alternative data source, mp4 video
    # cap = cv2.VideoCapture('/path/to/video.mp4')
    # cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    # while (cap.isOpened()):
    #     ret, img = cap.read()

    documents = collection.find({}).limit(100)
    for doc in documents:
        decoded_img = np.frombuffer(doc["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
        
        img, roi = resize_img(img, args.img_width, args.img_height, args.offset_bottom)

        if is_tf_lite:
            img_input = img
            interpreter.set_tensor(input_details[0]['index'], [img_input])
            #input_shape = input_details[0]['shape']
            start_time = time.time()
            interpreter.invoke()
            elapsed_time = time.time() - start_time
            # The function `get_tensor()` returns a copy of the tensor data. Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
        else:
            img_arr = np.array([img])
            start_time = time.time()
            raw_result = model.predict(img_arr)
            elapsed_time = time.time() - start_time
            output_data = raw_result[0]

        print(str(elapsed_time) + " s")

        # TODO: Get all stuff and display it
        # resized_semseg_img = cv2.resize(semseg_img, (args.img_width, args.img_height))
        # surface_input_img = pygame.surfarray.make_surface(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
        # display.blit(surface_input_img, (0, 0))
        # surface_semseg_mask = pygame.surfarray.make_surface(cv2.cvtColor(resized_semseg_img, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
        # display.blit(surface_semseg_mask, (0, args.img_height * 2))
        pygame.display.flip()

        # wait till space is pressed
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_SPACE:
                break

    pygame.quit()
