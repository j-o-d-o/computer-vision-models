import numpy as np
import cv2
from dataclasses import dataclass
from data.od_spec import OD_CLASS_MAPPING


@dataclass
class Roi:
    # offsets are pre scaling!
    offset_top: int = 0
    offset_bottom: int = 0
    offset_left: int = 0
    offset_right: int = 0
    scale: float = 1.0


def resize_img(img: np.ndarray, goal_width: int, goal_height: int, offset_bottom: int = 0, interpolation: int = cv2.INTER_LINEAR) -> (np.ndarray, Roi):
    """
    Resize image in a way that it fits the params, the default cropping will take delta height from top and delta width
    from left and right border equally
    :param img: numpy img array (as used by cv2), note that it will also be changed in place
    :param goal_width: width the image should have after resizing
    :param goal_height: height the image should have after resizing
    :param offset_bottom: offset from bottom e.g. to cut away hood of car (in org image scale)
    :param interpolation: Interpolation which should be used, default is cv2.INTER_LINEAR
    :return: scaled and cropped image, roi data
    """
    roi = Roi()
    # Add or remove offset_bottom
    roi.offset_bottom = offset_bottom
    h, w = img.shape[:2]
    if roi.offset_bottom > 0:
        new_img = np.zeros((h+offset_bottom, w))
        new_img[:h,:] = img
        img = new_img
    else:
        img = img[:(h+roi.offset_bottom), :]
    
    h, w = img.shape[:2]

    curr_ratio = w / float(h)
    target_ratio = goal_width / float(goal_height)
    if curr_ratio > target_ratio:
        # cut delta width equally from left and right edge
        delta_width = int((target_ratio * h) - w)
        roi.offset_left += (delta_width // 2) + (delta_width % 2)
        roi.offset_right += delta_width // 2
        img = img[:, -roi.offset_left:(w+roi.offset_right)]
    else:
        # cut delta height from top
        roi.offset_top = int((w / target_ratio) - h)
        img = img[-roi.offset_top:h, :]
    unscaled_h, unscaled_w = img.shape[:2]
    roi.scale = goal_width / float(unscaled_w)
    img = cv2.resize(img, (goal_width, goal_height), interpolation=interpolation)
    return img, roi

def to_hex(array):
    """
    Convert 3 channel representation to single hex channel
    :param array: 2d image with rgb channels
    :return: 2d image with single hex value
    """
    img = np.asarray(img, dtype='uint32')
    return (img[:, :, 0] << 16) + (img[:, :, 1] << 8) + img[:, :, 2]

def to_3channel(raw_mask_output, class_mapping):
    """
    Convert a mask with one hot encoded class mapping to a rgb coded colour image
    :param raw_mask_output: 2d img as mask outpu with channels > number of classes, also classes are expected to be first in the channels
    :param class_mapping: A ordered dict of class with key: string and value: (r, g, b)
    :return: 2d rgba image that encodes the class scores for each pixel
    """
    array = np.zeros((raw_mask_output.shape[0] * raw_mask_output.shape[1] * 3), dtype='uint8')
    flattened_arr = raw_mask_output.reshape((-1, raw_mask_output.shape[2]))
    for i, one_hot_encoded_arr in enumerate(flattened_arr):
        # find index of highest value in the one_hot_encoded_arr
        nb_classes = len(class_mapping)
        cls_idx = np.argmax(one_hot_encoded_arr[:nb_classes])
        max_val = one_hot_encoded_arr[cls_idx]
        assert(max_val >= 0.0 and max_val <= 1.0)
        # convert index to hex value
        cls_colour = list(class_mapping.items())[int(round(cls_idx))][1]
        # fill new array with BGR values
        new_i = i * 3
        array[new_i] = cls_colour[0] * max_val
        array[new_i + 1] = cls_colour[1] * max_val
        array[new_i + 2] = cls_colour[2] * max_val

    return array.reshape((raw_mask_output.shape[0], raw_mask_output.shape[1], 3))
