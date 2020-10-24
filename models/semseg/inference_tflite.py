import os
import cv2
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from models.semseg.processor import ProcessImages

gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(gpus[0], True)

MODEL_PATH = "/home/jodo/trained_models/semseg_16-08-2020-13-16-55/tf_model_36/model_quant.tflite"
IMG_WIDTH = 320
IMG_HEIGHT = 130
OFFSET_TOP = 60 # from original size (640x380)
CLASS_MAPPING = OrderedDict([
    ("road", 0x202040),
    ("lane_markings", 0x0000ff),
    ("undriveable", 0x608080),
    ("movable", 0x66ff00),
    ("ego_car", 0xff00cc),
])


def to_3channel(raw_semseg_img):
    array = np.zeros((raw_semseg_img.shape[0] * raw_semseg_img.shape[1] * 3), dtype='uint8')
    flattened_arr = raw_semseg_img.reshape((-1, len(CLASS_MAPPING)))
    for i, one_hot_encoded_arr in enumerate(flattened_arr):
        # find index of highest value in the one_hot_encoded_arr
        cls_idx = np.argmax(one_hot_encoded_arr)
        # convert index to hex value
        hex_val = int(list(CLASS_MAPPING.items())[int(round(cls_idx))][1])
        # fill new array with BGR values
        new_i = i * 3
        array[new_i] = (hex_val & 0xFF0000) >> 16
        array[new_i + 1] = (hex_val & 0x00FF00) >> 8
        array[new_i + 2] = (hex_val & 0x0000FF)

    return array.reshape((raw_semseg_img.shape[0], raw_semseg_img.shape[1], 3))


if __name__ == "__main__":
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    DIR_PATH = '/home/jodo/ILONA/visu/export'
    img_list = [file for file in os.listdir(DIR_PATH) if file.endswith('.png')]
    for img_name in img_list:
        png_img_path = DIR_PATH + "/" + img_name
        img = cv2.imread(png_img_path, cv2.IMREAD_COLOR)
        img, roi = ProcessImages.resize_img(img, IMG_WIDTH, IMG_HEIGHT, OFFSET_TOP)
        # test if col first or row first is mixed up
        #swapped = np.swapaxes(img, 0, 1)
        #swapped = swapped.reshape((-1, 3))
        #swapped = swapped.reshape((130, 320, 3))
        #img = swapped

        img = img.astype(np.float32)

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        print(input_shape)
        interpreter.set_tensor(input_details[0]['index'], [img])
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #swapped = np.swapaxes(output_data[0], 0, 1)
        #swapped = swapped.reshape((-1, 5))
        #swapped = swapped.reshape((130, 320, 5))
        #output_data[0] = swapped

        semseg_img = to_3channel(output_data[0])

        img = img.astype(np.uint8)
        cv2.imshow("Input Image", img)
        cv2.imshow("Semseg Image", semseg_img)
        cv2.waitKey(0)
