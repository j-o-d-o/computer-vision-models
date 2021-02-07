import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import cv2
import argparse
from pymongo import MongoClient
from common.utils import resize_img, tflite_convert


def create_dataset(input_shape):
    print("Resize Input to: " + str(input_shape))
    dataset = []
    
    # Create sample dataset for post training quantization
    client = MongoClient("mongodb://localhost:27017")
    collection = client["semseg"]["comma10k"]
    documents = collection.find({}).limit(600)

    documents_list = list(documents)
    assert(len(documents_list) > 0)
    for doc in documents_list:
        decoded_img = np.frombuffer(doc["img"], np.uint8)
        img = cv2.imdecode(decoded_img, cv2.IMREAD_COLOR)
        img, roi = resize_img(img, input_shape[2], input_shape[1], offset_bottom=-242)
        dataset.append(np.array([img], dtype=np.float32))

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize tensorflow model to tflite and edgetpu")
    parser.add_argument("--model_path", type=str, help="Path to a tensorflow model folder (SaveModel and H5 format supported)")
    parser.add_argument("--quantize", action="store_true", help="Quantize model using input data")
    parser.add_argument("--compile_edge_tpu", action="store_true", help="Compile TFLite model also for EdgeTpu")
    args = parser.parse_args()

    # args.compile_edge_tpu = True
    # args.quantize = True
    # args.compile_edge_tpu = True
    # args.model_path = "/home/computer-vision-models/trained_models/semseg_comma10k_augment_2021-02-06-062841/tf_model_40/keras.h5"

    model = tf.keras.models.load_model(args.model_path, compile=False)

    save_dir = args.model_path
    if save_dir.lower().endswith(".h5"):
        save_dir = os.path.split(save_dir)[0]

    tflite_convert(model, save_dir, args.quantize, args.compile_edge_tpu, create_dataset(model.input.shape))
