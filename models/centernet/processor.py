import numpy as np
import cv2
import math
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.od_spec import OD_CLASS_MAPPING, OD_CLASS_IDX
from models.centernet.params import CenternetParams
import albumentations as A


class ProcessImages(IPreProcessor):
    def __init__(self, params: CenternetParams):
        self.params = params
        self.SHOW_DEBUG_IMG = False # showing the input image including all object's 2d information drawn

    def fill_heatmap(self, ground_truth, cls_idx, center_x, center_y, width, height, mask_width, mask_height, peak = 1.0):
        # create the heatmap with a gausian distribution for lower loss in the area of each object
        min_x = max(0, center_x - int(width // 2))
        max_x = min(mask_width, center_x + int(width // 2))
        min_y = max(0, center_y - int(height // 2))
        max_y = min(mask_height, center_y + int(height // 2))
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                var_width = math.pow(((self.params.VARIANCE_ALPHA * width) / (6 * self.params.R)), 2)
                var_height = math.pow(((self.params.VARIANCE_ALPHA * height) / (6 * self.params.R)), 2)
                weight_width = math.pow((x - center_x), 2) / (2 * var_width)
                weight_height = math.pow((y - center_y), 2) / (2 * var_height)
                var_weight = math.exp(-(weight_width + weight_height))
                ground_truth[y][x][cls_idx] = max(var_weight * peak, ground_truth[y][x][cls_idx])

    def calc_img_data(self, obj, roi, mask_width, mask_height):
        # get center points and their offset
        scaled_box2d = ((np.asarray(obj["box2d"]) + [roi.offset_left, roi.offset_top, 0, 0]) * roi.scale) / float(self.params.R)
        x, y, width, height = scaled_box2d
        center_x_float = x + (float(width) / 2.0)
        center_y_float = y + (float(height) / 2.0)
        # TODO: When the fullbox is outside of the image, readjust the fullbox and the center accordingly
        #       if fullbox is completely out of image, remove the track
        center_x = max(0, min(mask_width - 1, int(center_x_float))) # index needs to be int and within mask range
        center_y = max(0, min(mask_height - 1, int(center_y_float))) # index needs to be int and within mask range
        loc_off_x = center_x_float - center_x
        loc_off_y = center_y_float - center_y

        # fullbox (scale back to input image size)
        fullbox_width = width * self.params.R
        fullbox_height = height * self.params.R

        valid_l_shape = False
        bottom_left_off = bottom_right_off = bottom_center_off = center_height = None
        if obj["box3d_valid"]:
            # find bottom_left and bottom_right points
            offsetsBox3d = [roi.offset_left, roi.offset_top] * 8
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
            center_height = bottom_center[1] - top_center[1]

            # convert the bottom points to offsets
            cp = np.asarray([center_x_float, center_y_float]) * float(self.params.R)
            bottom_left_off = bottom_left - cp
            bottom_right_off = bottom_right - cp
            bottom_center_off = bottom_center - cp

            # Check if we got a valid l-shape
            valid_l_shape = True
            off_to_center = [(self.params.R * mask_width * 0.5), (self.params.R * mask_height * 0.5)]
            threshold = self.params.R * mask_width * 2
            if (sum(abs(bottom_left_off - off_to_center)) > threshold):
                valid_l_shape = False
            if (sum(abs(bottom_right_off - off_to_center)) > threshold):
                valid_l_shape = False
            if (sum(abs(bottom_center_off - off_to_center)) > threshold):
                valid_l_shape = False

        return center_x, center_y, loc_off_x, loc_off_y, fullbox_width, fullbox_height, valid_l_shape, bottom_left_off, bottom_right_off, bottom_center_off, center_height

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data_unscaled = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi = resize_img(input_data_unscaled, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)
        
        # some augmentation
        transform = A.Compose([
            A.IAAAdditiveGaussianNoise(p=0.2),
            A.OneOf([
                A.IAASharpen(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ] , p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.RandomGamma(p=1.0),
                A.CLAHE(p=1.0)
            ],p=0.3),
            A.OneOf([
                A.RandomFog(p=1.0),
                A.RandomRain(p=1.0),
                A.RandomShadow(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomSunFlare(p=1.0)
            ], p=0.05),
        ])
        transformed = transform(image=input_data)
        input_data = transformed["image"]
        
        input_data = input_data.astype(np.float32)
        if self.SHOW_DEBUG_IMG:
            debug_img = input_data.copy().astype(np.uint8)

        # pipe params in case centertracker is used
        piped_params["roi"] = roi 
        piped_params["gt_2d_info"] = []

        # Add ground_truth mask/heatmap
        mask_width = self.params.INPUT_WIDTH // self.params.R
        mask_height = self.params.INPUT_HEIGHT // self.params.R
        mask_channels = self.params.mask_channels()
        ground_truth = np.zeros((mask_height, mask_width, mask_channels))

        # some debugging
        for obj in raw_data["objects"]:
            # create 2D information (see od_spec.py for format):
            # heatmap values: center_x, center_y
            # regression values: bottom_left, bottom_center, bottom_right, height, offset_x, offset_y
            center_x, center_y, loc_off_x, loc_off_y, width, height, valid_l_shape, bottom_left_off, bottom_right_off, bottom_center_off, center_height \
                = self.calc_img_data(obj, roi, mask_width, mask_height)
            piped_params["gt_2d_info"].append([center_x, center_y, width, height])
            
            if self.params.REGRESSION_FIELDS["l_shape"].active and not valid_l_shape:
                # TODO: Add this object to the ignore areas
                pass
            else:
                # fill ground truth mask for all active fields
                gt_center = ground_truth[center_y][center_x][:]
                if self.params.REGRESSION_FIELDS["r_offset"].active:
                    gt_center[self.params.start_idx("r_offset"):self.params.end_idx("r_offset")] = [loc_off_x, loc_off_y]
                if self.params.REGRESSION_FIELDS["fullbox"].active:
                    gt_center[self.params.start_idx("fullbox"):self.params.end_idx("fullbox")] = [width, height]
                if self.params.REGRESSION_FIELDS["l_shape"].active:
                    gt_center[self.params.start_idx("l_shape"):self.params.end_idx("l_shape")] = [*bottom_left_off, *bottom_right_off, *bottom_center_off, center_height]
                if self.params.REGRESSION_FIELDS["3d_info"].active:
                    radial_dist = math.sqrt(obj["x"] ** 2 + obj["y"] ** 2 + obj["z"] ** 2)
                    gt_center[self.params.start_idx("3d_info"):self.params.end_idx("3d_info")] = [radial_dist, obj["orientation"], obj["width"], obj["height"], obj["length"]]

                # create the heatmap with a gausian distribution for lower loss in the area of each object
                self.fill_heatmap(ground_truth, OD_CLASS_IDX[obj["obj_class"]], center_x, center_y, width, height, mask_width, mask_height)

                if self.SHOW_DEBUG_IMG:
                    # center
                    cp = np.array([center_x + loc_off_x, center_y + loc_off_y]) * float(self.params.R)
                    cv2.circle(debug_img, (int(cp[0]), int(cp[1])), 4, (0, 0, 255), 2)
                    # fullbox
                    top_left = (int(cp[0] - (width / 2.0)), int(cp[1] - (height / 2.0)))
                    bottom_right = (int(cp[0] + (width / 2.0)), int(cp[1] + (height / 2.0)))
                    cv2.rectangle(debug_img, top_left, bottom_right, (255, 0, 0), 1)
                    # l-shape
                    if valid_l_shape:
                        cv2.line(debug_img, tuple((bottom_left_off + cp).astype(np.int32)), tuple((bottom_center_off + cp).astype(np.int32)), (0, 255, 0) , 1) 
                        cv2.line(debug_img, tuple((bottom_center_off + cp).astype(np.int32)), tuple((bottom_right_off + cp).astype(np.int32)), (0, 255, 0) , 1) 
                        cv2.line(debug_img, tuple((bottom_center_off + cp).astype(np.int32)), (int((bottom_center_off + cp)[0]), int((bottom_center_off + cp)[1] - center_height)), (0, 255, 0) , 1)

        # TODO: Add ignore_flags to output

        if self.SHOW_DEBUG_IMG:
            cv2.imshow("Debug Test", debug_img)
            cv2.waitKey(0)

        return raw_data, input_data, ground_truth, piped_params
