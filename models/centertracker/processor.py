import numpy as np
import cv2
import math
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.od_spec import OD_CLASS_MAPPING, OD_CLASS_IDX
from models.centernet.params import Params

SHOW_DEBUG_IMG = False # showing the input image including all object's 2d information drawn


class ProcessImages(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data_unscaled = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi = resize_img(input_data_unscaled, Params.INPUT_WIDTH, Params.INPUT_HEIGHT, offset_bottom=Params.OFFSET_BOTTOM)
        input_data = input_data.astype(np.float32)
        if SHOW_DEBUG_IMG:
            debug_img = input_data.copy().astype(np.uint8)

        # Add ground_truth mask/heatmap
        mask_width = Params.INPUT_WIDTH // Params.R
        mask_height = Params.INPUT_HEIGHT // Params.R
        nb_classes = len(OD_CLASS_MAPPING)
        mask_channels = nb_classes + 14 # nb_classes + nb_regression_parameters
        ground_truth = np.zeros((mask_height, mask_width, mask_channels))

        # some debugging
        for obj in raw_data["objects"]:
            # create 2D information (see od_spec.py for format):
            # heatmap values: center_x, center_y
            # regression values: bottom_left, bottom_center, bottom_right, height, offset_x, offset_y

            # get center points and their offset
            scaled_box2d = ((np.asarray(obj["box2d"]) + [roi.offset_left, roi.offset_top, 0, 0]) * roi.scale) / float(Params.R)
            x, y, width, height = scaled_box2d
            center_x_float = x + (float(width) / 2.0)
            center_y_float = y + (float(height) / 2.0)
            center_x = max(0, min(mask_width - 1, int(center_x_float))) # index needs to be int and within mask range
            center_y = max(0, min(mask_height - 1, int(center_y_float))) # index needs to be int and within mask range
            offset_x = center_x_float - center_x
            offset_y = center_y_float - center_y

            # find bottom_left and bottom_right points
            offsetsBox3d = [roi.offset_left, roi.offset_bottom] * 8
            box3d = (np.asarray(obj["box3d"]) + offsetsBox3d) * roi.scale
            top_points = np.asarray([box3d[0:2], box3d[6:8], box3d[8:10], box3d[14:]])
            bottom_points = np.asarray([box3d[2:4], box3d[4:6], box3d[10:12], box3d[12:14]])
            min_val = np.argmin(bottom_points, axis=0) # min values in x and y direction
            max_val = np.argmax(bottom_points, axis=0) # max value in x and y direction
            bottom_left = bottom_points[min_val[0]]
            bottom_right = bottom_points[max_val[0]]
            # from the two remaning bottom points find the max y value for the bottom_center point
            mask = np.zeros((4, 2), dtype=bool)
            mask[[min_val[0], max_val[0]]] = True
            remaining_points = np.ma.array(bottom_points, mask=mask)
            max_val = np.argmax(remaining_points, axis=0) # max value in x and y direction
            bottom_center = remaining_points[max_val[1]]
            # take the top point of the found center as height in pixel
            top_center = top_points[max_val[1]]
            height_px = bottom_center[1] - top_center[1]
            # calc offset
            center_x_float = scaled_box2d[0] + (float(width) / 2.0)
            center_y_float = scaled_box2d[1] + (float(height) / 2.0)
            center_x = max(0, min(mask_width - 1, int(center_x_float))) # index needs to be int and within mask range
            center_y = max(0, min(mask_height - 1, int(center_y_float))) # index needs to be int and within mask range
            offset_x = center_x_float - center_x
            offset_y = center_y_float - center_y

            # fill all of it into the ground_truth mask
            gt_test = ground_truth[center_y][center_x]
            gt_regression = ground_truth[center_y][center_x][nb_classes:]
            gt_regression[0] = offset_x
            gt_regression[1] = offset_y
            gt_regression[2] = bottom_left[0]
            gt_regression[3] = bottom_left[1]
            gt_regression[4] = bottom_center[0]
            gt_regression[5] = bottom_center[1]
            gt_regression[6] = bottom_right[0]
            gt_regression[7] = bottom_right[1]
            gt_regression[8] = height_px
            gt_regression[9] = math.sqrt(obj["x"] ** 2 + obj["y"] ** 2 + obj["z"] ** 2)
            gt_regression[10] = obj["orientation"]
            gt_regression[11] = obj["width"]
            gt_regression[12] = obj["height"]
            gt_regression[13] = obj["length"]

            # TODO: Add ignore_flags to output

            # create the heatmap with a gausian distribution for lower loss in the area of each object
            cls_idx = OD_CLASS_IDX[obj["obj_class"]]
            min_x = max(0, center_x - int(width // 2))
            max_x = min(mask_width, center_x + int(width // 2))
            min_y = max(0, center_y - int(height // 2))
            max_y = min(mask_height, center_y + int(height // 2))
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    var_width = math.pow(((Params.VARIANCE_ALPHA * width) / (6 * Params.R)), 2)
                    var_height = math.pow(((Params.VARIANCE_ALPHA * height) / (6 * Params.R)), 2)
                    weight_width = math.pow((x - center_x), 2) / (2 * var_width)
                    weight_height = math.pow((y - center_y), 2) / (2 * var_height)
                    weight = math.exp(-(weight_width + weight_height))
                    ground_truth[y][x][cls_idx] = max(weight, ground_truth[y][x][cls_idx])

            if SHOW_DEBUG_IMG:
                cv2.line(debug_img, tuple(bottom_left.astype(np.int32)), tuple(bottom_center.astype(np.int32)), (0, 255, 0) , 1) 
                cv2.line(debug_img, tuple(bottom_center.astype(np.int32)), tuple(bottom_right.astype(np.int32)), (0, 255, 0) , 1) 
                cv2.line(debug_img, tuple(bottom_center.astype(np.int32)), (int(bottom_center[0]), int(bottom_center[1] - height_px)), (0, 255, 0) , 1)

        if SHOW_DEBUG_IMG:
            cv2.imshow("Debug Test", debug_img)
            cv2.waitKey(0)

        return raw_data, input_data, ground_truth, piped_params
