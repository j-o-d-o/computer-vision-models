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
        input_data, roi = resize_img(input_data, Params.INPUT_WIDTH, Params.INPUT_HEIGHT, offset_bottom=Params.OFFSET_BOTTOM)
        input_data = input_data.astype(np.float32)

        # Add ground_truth mask/heatmap
        mask_width = Params.INPUT_WIDTH // Params.R
        mask_height = Params.INPUT_HEIGHT // Params.R
        nb_classes = len(OD_CLASS_MAPPING)
        mask_channels = nb_classes + 4 # nb_classes + offset_x + offset_y + width + height
        ground_truth = np.zeros((mask_height, mask_width, mask_channels))

        for obj in raw_data["objects"]:
            # Filter boxes that shouldnt be used for training
            if (obj["box2d"][2] * roi.scale) < Params.MIN_BOX_WIDTH or (obj["box2d"][3] * roi.scale) < Params.MIN_BOX_HEIGHT:
                # TODO: If box is filtered out, it should be added to the ignore area!
                continue

            # Boxe's location and size are scaled to size of output_mask
            #TODO: Offsets are not accounted for!
            scaled_box2d = (np.asarray(obj["box2d"]) * roi.scale) / float(Params.R)
            width = scaled_box2d[2]
            height = scaled_box2d[3]
            center_x_float = scaled_box2d[0] + (float(width) / 2.0)
            center_y_float = scaled_box2d[1] + (float(height) / 2.0)
            center_x = max(0, min(mask_width - 1, int(center_x_float))) # index needs to be int and within mask range
            center_y = max(0, min(mask_height - 1, int(center_y_float))) # index needs to be int and within mask range
            offset_x = center_x_float - center_x
            offset_y = center_y_float - center_y
            # Fill offset x and y
            ground_truth[center_y][center_x][nb_classes + 0] = offset_x
            ground_truth[center_y][center_x][nb_classes + 1] = offset_y
            # Fill width and height at keypoint
            ground_truth[center_y][center_x][nb_classes + 2] = width
            ground_truth[center_y][center_x][nb_classes + 3] = height

            # TODO: Add ignore_flag

            # Fill an area that is the size of the object with a gausian distribution for lower loss in that area
            cls_idx = OD_CLASS_IDX[obj["obj_class"]]
            min_x = max(0, center_x - int(width // 2))
            max_x = min(mask_width, center_x + int(width // 2))
            min_y = max(0, center_y - int(height // 2))
            max_y = min(mask_height, center_y + int(height // 2))
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    varWidth = math.pow(((Params.VARIANCE_ALPHA * width) / (6 * Params.R)), 2)
                    varHeight = math.pow(((Params.VARIANCE_ALPHA * height) / (6 * Params.R)), 2)
                    weight_width = math.pow((x - center_x), 2) / (2 * varWidth)
                    weight_height = math.pow((y - center_y), 2) / (2 * varHeight)
                    weight = math.exp(-weight_width - weight_height)
                    ground_truth[y][x][cls_idx] = max(weight, ground_truth[y][x][cls_idx])

        return raw_data, input_data, ground_truth, piped_params
