import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.label_spec import OD_CLASS_MAPPING, OD_CLASS_IDX
from models.centernet.params import CenternetParams
import albumentations as A


class ProcessImages(IPreProcessor):
    def __init__(self, params: CenternetParams, start_augmentation: int = None, show_debug_img: bool = False):
        self.params = params
        self.start_augmentation = start_augmentation
        self.show_debug_img = show_debug_img

    def fill_heatmap(self, ground_truth, weights, cls_idx, center_x, center_y, width, height, mask_width, mask_height, peak = 1.0):
        max_dim = max(width, height)
        reduce_weight = 1.0 - ((min(20.0, max_dim) / 16.0) - 0.25)
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
                weights[y][x] = min(weights[y][x], 1.0 - (reduce_weight * var_weight))

    def calc_img_data(self, box2d, box3d, mask_width, mask_height):
        # get center points and their offset
        scaled_box2d = np.asarray(box2d) / float(self.params.R)
        x, y, width, height = scaled_box2d
        center_x_float = x + (float(width) / 2.0)
        center_y_float = y + (float(height) / 2.0)
        center_x = max(0, min(mask_width - 1, int(center_x_float))) # index needs to be int and within mask range
        center_y = max(0, min(mask_height - 1, int(center_y_float))) # index needs to be int and within mask range
        center = [center_x, center_y]
        loc_off = [center_x_float - center_x, center_y_float - center_y]

        valid_l_shape = False
        l_shape = []
        if box3d is not None:
            bottom_left_off = bottom_right_off = bottom_center_off = center_height = None
            # find bottom_left and bottom_right points
            box3d = np.asarray(box3d)
            top_points = np.asarray([box3d[0], box3d[3], box3d[4], box3d[7]])
            bottom_points = np.asarray([box3d[1], box3d[2], box3d[5], box3d[6]])
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

            l_shape = [bottom_left_off, bottom_center_off, bottom_right_off, center_height]

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

        return center, loc_off, valid_l_shape, l_shape

    def augment(self, img1, bbox1 = [], keypoints1 = [], img0 = None, bbox0 = [], keypoints0 = [], do_img_aug = True, do_affine_aug = False, img1_to_img0 = False):
        if img0 is None:
            img0 = img1.copy()

        keypoints = keypoints1 + keypoints0
        bboxes = bbox1 + bbox0

        # Afine augmentations
        # --------------------------------------
        if do_affine_aug:
            afine_transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.4),
                    A.OneOf([
                        A.GridDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                        A.ElasticTransform(interpolation=cv2.INTER_NEAREST, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                        A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, shift_limit=0.035, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                        A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    ], p=0.5),
                ],
                additional_targets={"img0": "image"},
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format='coco', label_fields=[]), # coco format: [x-min, y-min, width, height]
            )
            transformed = afine_transform(image=img1, img0=img0, keypoints=keypoints, bboxes=bboxes)
            img1 = transformed["image"]
            img0 = transformed["img0"]
            transformed_keypoints = transformed['keypoints']
            keypoints1 = transformed_keypoints[:len(keypoints1)]
            keypoints0 = transformed_keypoints[len(keypoints1):]
            transformed_bboxes = transformed['bboxes']
            bbox1 = transformed_bboxes[:len(bbox1)]
            bbox0 = transformed_bboxes[len(bbox1):]

        # Translate img0 + bbox0/keypoints0 for single track centernet
        # --------------------------------------
        if img1_to_img0:
            transform = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, always_apply=True, border_mode=cv2.BORDER_CONSTANT)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                bbox_params=A.BboxParams(format='coco', label_fields=[])
            )
            keypoints = list(np.array(piped_params["gt_2d_info"])[:,:2] * self.params.R)
            transformed = transform(image=img1, keypoints=keypoints1, bboxes=bboxes1)
            img0 = transformed["image"]
            bboxes0 = transformed['bboxes']
            keypoints0 = transformed['keypoints']

        # Augmentation for images
        # --------------------------------------
        if do_img_aug:
            transform = A.Compose([
                A.IAAAdditiveGaussianNoise(p=0.02),
                A.OneOf([
                    A.IAASharpen(p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ] , p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomGamma(p=1.0),
                ], p=0.6),
                A.OneOf([
                    A.RandomFog(p=1.0),
                    A.RandomRain(p=1.0),
                    A.RandomShadow(p=1.0),
                    A.RandomSnow(p=1.0)
                ], p=0.02),
            ], additional_targets={"img0": "image"})
            transformed = transform(image=img1, img0=img0)
            img1 = transformed["image"]
            img0 = transformed["img0"]

        return img1, bbox1, keypoints1, img0, bbox0, keypoints0

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # img1 = image at t+1
        # img0 = image at t
        # images are expected to be in INPUT_HEIGHT and INPUT_WIDTH already
        img0 = None
        img1 = None
        objects0 = []
        objects1 = []
        bbox0 = []
        bbox1 = []
        ignore_areas = []
        keypoints0 = []
        keypoints1 = []

        if self.params.REGRESSION_FIELDS["track_offset"].active:
            # TODO: Do center tracker part thingy
            pass
        else:
            assert(not isinstance(raw_data, list) and "Why get 2 images from database if no track offset is regressed?")
            img1 = cv2.imdecode(np.frombuffer(raw_data["img"], np.uint8), cv2.IMREAD_COLOR)
            objects1 = raw_data["objects"]

        # Get bboxes and keypoints for objects in format the augmentation needs
        for obj in objects1:
            bbox1.append(obj["box2d"])
            kp = np.ones([16])
            if obj["box3d_valid"]:
                kp = obj["box3d"]
            keypoints1 += list(np.array(kp).reshape(-1, 2))

        # TODO: In case of track_offset regression and only one image, transform img1 -> img0
        # TODO: Add keypoints and bboxes of img0 if available
        # TODO: Add ignore areas to augmentation
        # TODO: Need to clip bboxes and keypoints for affine agumentation...
        do_img_aug = True if self.start_augmentation is not None and piped_params["epoch"] >= self.start_augmentation[0] else False
        do_affine_aug = True if self.start_augmentation is not None and piped_params["epoch"] >= self.start_augmentation[1] else False
        img1, bbox1, keypoints1, img0, bbox0, keypoints0 = self.augment(img1, bbox1, keypoints1, do_img_aug=do_img_aug, do_affine_aug=do_affine_aug)

        # Create empty ground_truth mask/heatmap
        mask_width = self.params.INPUT_WIDTH // self.params.R
        mask_height = self.params.INPUT_HEIGHT // self.params.R
        mask_channels = self.params.mask_channels()
        heatmap1 = np.zeros((mask_height, mask_width, mask_channels), dtype=np.float32)
        weights = np.ones((mask_height, mask_width), dtype=np.float32) * 10.0

        if self.show_debug_img:
            debug_img0 = img0.copy().astype(np.uint8)

        # create ground truth heathmap for img0
        keypoints1 = list(np.asarray(keypoints1).reshape(-1, 8, 2)) # split into each object
        assert(len(keypoints1) == len(bbox1) and "Hmm, they should be the same length, better fix that bug.")

        for obj, bbox, keypoints in zip(objects1, bbox1, keypoints1):
            keypoints = keypoints if obj["box3d_valid"] else None
            center, loc_off, valid_l_shape, l_shape = self.calc_img_data(bbox, keypoints, mask_width, mask_height)

            width = bbox[2]
            height = bbox[3]

            if self.params.REGRESSION_FIELDS["l_shape"].active and not valid_l_shape:
                ignore_areas.append(bbox)
            else:
                # fill ground truth mask for all active fields
                gt_center = heatmap1[center[1]][center[0]][:]
                if self.params.REGRESSION_FIELDS["r_offset"].active:
                    gt_center[self.params.start_idx("r_offset"):self.params.end_idx("r_offset")] = loc_off
                if self.params.REGRESSION_FIELDS["fullbox"].active:
                    gt_center[self.params.start_idx("fullbox"):self.params.end_idx("fullbox")] = [width, height]
                if self.params.REGRESSION_FIELDS["l_shape"].active:
                    gt_center[self.params.start_idx("l_shape"):self.params.end_idx("l_shape")] = [*l_shape[0], *l_shape[1], *l_shape[2], l_shape[3]]
                if self.params.REGRESSION_FIELDS["3d_info"].active:
                    radial_dist = math.sqrt(obj["x"] ** 2 + obj["y"] ** 2 + obj["z"] ** 2)
                    gt_center[self.params.start_idx("3d_info"):self.params.end_idx("3d_info")] = [radial_dist, obj["orientation"], obj["width"], obj["height"], obj["length"]]

                # create the heatmap with a gausian distribution for lower loss in the area of each object
                self.fill_heatmap(heatmap1, weights, OD_CLASS_IDX[obj["obj_class"]], center[0], center[1], width, height, mask_width, mask_height)

                if self.show_debug_img:
                    # center
                    cp = np.array([center[0] + loc_off[0], center[1] + loc_off[1]]) * float(self.params.R)
                    cv2.circle(debug_img0, (int(cp[0]), int(cp[1])), 4, (0, 0, 255), 2)
                    # fullbox
                    top_left = (int(cp[0] - (width / 2.0)), int(cp[1] - (height / 2.0)))
                    bottom_right = (int(cp[0] + (width / 2.0)), int(cp[1] + (height / 2.0)))
                    cv2.rectangle(debug_img0, top_left, bottom_right, (255, 0, 0), 1)
                    # l-shape
                    if valid_l_shape:
                        cv2.line(debug_img0, tuple((l_shape[0] + cp).astype(np.int32)), tuple((l_shape[1] + cp).astype(np.int32)), (0, 255, 0) , 1) 
                        cv2.line(debug_img0, tuple((l_shape[1] + cp).astype(np.int32)), tuple((l_shape[2] + cp).astype(np.int32)), (0, 255, 0) , 1) 
                        cv2.line(debug_img0, tuple((l_shape[1] + cp).astype(np.int32)), (int((l_shape[1] + cp)[0]), int((l_shape[1] + cp)[1] - l_shape[3])), (0, 255, 0) , 1)

        for ignore_area in ignore_areas:
            start_y = int(ignore_area[1])
            end_y = int(ignore_area[1] + ignore_area[3])
            start_x = int(ignore_area[0])
            end_x = int(ignore_area[0] + ignore_area[2])
            weights[start_y:end_y, start_x:end_x] = 0.0

        # TODO: Create input heatmap for img0 if track offset is regressed

        if self.show_debug_img:
            f, (ax1) = plt.subplots(1, 1)
            ax1.imshow(cv2.cvtColor(debug_img0, cv2.COLOR_BGR2RGB))
            plt.show()

        weights = np.where(weights == 10.0, 1.0, weights)
        input_data = [img1.astype(np.float32)]
        ground_truth = [heatmap1, weights]

        return raw_data, input_data, ground_truth, piped_params
