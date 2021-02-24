import numpy as np
import cv2
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from data.label_spec import SEMSEG_CLASS_MAPPING
from models.multitask import MultitaskParams
import albumentations as A
from numba.typed import List
from models.semseg.processor import to_hex, to_categorical, hex_to_one_hot


class ProcessImages(IPreProcessor):
    def __init__(self, params: MultitaskParams):
        self.params: MultitaskParams = params

    def augment(self, img, mask, depth, do_afine_transform: bool = False):
        if do_afine_transform:
            afine_transform = A.Compose([
                A.HorizontalFlip(p=0.4),
                A.OneOf([
                    A.GridDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ElasticTransform(interpolation=cv2.INTER_NEAREST, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                    A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
                ], p=0.5),
            ], additional_targets={'mask': 'image', 'depth': 'image'})
            afine_transformed = afine_transform(image=img, mask=mask, depth=depth)
            img = afine_transformed["image"]
            mask = afine_transformed["mask"]
            depth = afine_transform["depth"]

        transform = A.Compose([
            A.IAAAdditiveGaussianNoise(p=0.05),
            A.OneOf([
                A.IAASharpen(p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ] , p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.RandomGamma(p=1.0),
            ], p=0.5),
            A.OneOf([
                A.RandomFog(p=1.0),
                A.RandomRain(p=1.0),
                A.RandomShadow(p=1.0),
                A.RandomSnow(p=1.0)
            ], p=0.05),
        ])
        transformed = transform(image=img)
        img = transformed["image"]

        return img, mask

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi_img = resize_img(input_data, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)
        piped_params["roi_img"] = roi_img

        semseg_img = []
        depth_img = []
        pos_mask = np.ones((self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH))
        depth_img = np.zeros((self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH))
        semseg_img = np.zeros((self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, len(SEMSEG_CLASS_MAPPING)))
        semseg_valid = False
        depth_valid = False

        # Add ground_truth mask
        if raw_data["mask"] is not None:
            mask_encoded = np.frombuffer(raw_data["mask"], np.uint8)
            semseg_img = cv2.imdecode(mask_encoded, cv2.IMREAD_COLOR)
            semseg_img, _ = resize_img(semseg_img, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)
            # one hot encode based on class mapping from semseg spec
            semseg_img = to_hex(semseg_img) # convert 3 channel representation to single hex channel
            colours = List()
            for _, colour in list(SEMSEG_CLASS_MAPPING.items()):
                colours.append((colour[0] << 16) + (colour[1] << 8) + colour[2])
            semseg_img, pos_mask = hex_to_one_hot(semseg_img, pos_mask, colours)
            semseg_img = to_categorical(semseg_img, len(SEMSEG_CLASS_MAPPING))
            semseg_valid = True

        if raw_data["depth"] is not None:
            depth_img = cv2.imdecode(np.frombuffer(raw_data["depth"], np.uint8), cv2.IMREAD_ANYDEPTH)
            depth_img, _ = resize_img(depth_img, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)
            depth_img = depth_img.astype(np.float32)
            depth_img /= 255.0
            depth_masking = np.where(depth_img > 0.01, 1.0, 0.0) 
            depth_img = np.clip(depth_img, 4.1, 130.0)
            depth_img = 22 * np.sqrt(depth_img - 4)
            depth_img *= depth_masking
            depth_valid = True

        input_data = input_data.astype(np.float32)
        ground_truth = [semseg_img, semseg_valid, depth_img, depth_valid, pos_mask]
        return raw_data, input_data, ground_truth, piped_params
