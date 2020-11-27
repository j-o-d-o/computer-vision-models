import numpy as np
import cv2
import math
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
        nb_classes = len(OD_CLASS_MAPPING)
        mask_channels = nb_classes + 2 # nb_classes + width + height
        ground_truth = np.zeros((mask_height, mask_width, mask_channels))
        # Add objects
        for obj in raw_data["objects"]:
            scaled_box2d = (np.asarray(obj["box2d"]) * roi.scale) // float(Params.R)
            width = int(scaled_box2d[2])
            height = int(scaled_box2d[3])
            center_x = int(scaled_box2d[0] + (width // 2.0))
            center_y = int(scaled_box2d[1] + (height // 2.0))
            # Fill width and height at keypoint
            ground_truth[center_y][center_x][nb_classes] = width
            ground_truth[center_y][center_x][nb_classes + 1] = height
            # TODO: Add offset_x, offset_y, ignore_flag
            # Fill an area that is half the size of the object width and height with a gausian distribution for lower loss in that area
            cls_idx = OD_CLASS_IDX[obj["obj_class"]]
            min_x = max(0, center_x - (width // 4))
            max_x = min(mask_width, center_x + (width // 4))
            min_y = max(0, center_y - (height // 4))
            max_y = min(mask_height, center_y + (height // 4))
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    stdDevWidth = width * 0.05
                    stdDevHeight = height * 0.05
                    score = math.exp(-((math.pow(x - center_x, 2) + math.pow(y - center_y, 2)) / (math.pow(stdDevHeight, 2) + math.pow(stdDevWidth, 2))))
                    ground_truth[y][x][cls_idx] = max(score, ground_truth[y][x][cls_idx])

        return raw_data, input_data, ground_truth, piped_params
