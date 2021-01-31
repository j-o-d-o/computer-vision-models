import numpy as np
import cv2
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.semseg_spec import SEMSEG_CLASS_MAPPING
from models.semseg.params import SemsegParams


def key_to_index(value):
    idx = 0
    for cls, colour in SEMSEG_CLASS_MAPPING.items():
        hex_colour = (colour[0] << 16) + (colour[1] << 8) + colour[2]
        if value == hex_colour:
            return idx
        idx += 1
    assert(False and "colour does not exist: " + str(value))
    return -1

def to_hex(img):
    """
    Convert 3 channel representation to single hex channel
    :param array: 2d image with rgb channels
    :return: 2d image with single hex value
    """
    img = np.asarray(img, dtype='uint32')
    return (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]

class ProcessImages(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi_img = resize_img(input_data, SemsegParams.INPUT_WIDTH, SemsegParams.INPUT_HEIGHT, offset_bottom=SemsegParams.OFFSET_BOTTOM)
        input_data = input_data.astype(np.float32)
        piped_params["roi_img"] = roi_img

        # Add ground_truth mask
        mask_encoded = np.frombuffer(raw_data["mask"], np.uint8)
        mask_img = cv2.imdecode(mask_encoded, cv2.IMREAD_COLOR)
        mask_img, roi_img = resize_img(mask_img, SemsegParams.MASK_WIDTH, SemsegParams.MASK_HEIGHT, offset_bottom=SemsegParams.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)
        piped_params["roi_img"] = roi_img
        # one hot encode based on class mapping from SemsegParams
        mask_img = to_hex(mask_img) # convert 3 channel representation to single hex channel
        vfunc = np.vectorize(key_to_index)
        mask_img = vfunc(mask_img)

        nb_classes = len(SEMSEG_CLASS_MAPPING)
        ground_truth = to_categorical(mask_img, nb_classes)

        return raw_data, input_data, ground_truth, piped_params
