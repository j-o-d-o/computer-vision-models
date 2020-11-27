import numpy as np
import cv2
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.od_spec import OD_CLASS_MAPPING, OD_CLASS_IDX
from models.centernet_2d.params import Params


class ProcessImages(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi = resize_img(input_data, Params.INPUT_WIDTH, Params.INPUT_HEIGHT)
        input_data = input_data.astype(np.float32)

        # Add ground_truth mask/heatmap
        mask_width = Params.INPUT_WIDTH // Params.R
        mask_height = Params.INPUT_HEIGHT // Params.R
        # TODO: Add offset_x, offset_y, ignore_flag
        mask_channels = len(OD_CLASS_MAPPING) + 2 # nb_classes + width + height
        ground_truth = np.zeros((mask_height, mask_width, mask_channels))
        # Add objects
        for obj in raw_data["objects"]:
            idx = OD_CLASS_IDX[obj["obj_class"]]
            scaled_box2d = (np.asarray(obj["box2d"]) * roi.scale) // float(Params.R)
            width = int(scaled_box2d[2])
            height = int(scaled_box2d[3])
            center_x = int(scaled_box2d[0] + (width // 2.0))
            center_y = int(scaled_box2d[1] + (height // 2.0))
            ground_truth[center_y][center_x][idx] = 1.0
            ground_truth[center_y][center_x][idx + 1] = width
            ground_truth[center_y][center_x][idx + 1] = height

        return raw_data, input_data, ground_truth, piped_params
