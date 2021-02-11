import numpy as np
import cv2
from dataclasses import dataclass
from common.processors import IPreProcessor
from common.utils import resize_img
from models.dmds.params import DmdsParams
import albumentations as A
import matplotlib.pyplot as plt


class ProcessImages(IPreProcessor):
    def __init__(self, params: DmdsParams):
        self.params: DmdsParams = params

    def augment(self, img, mask):
        afine_transform = A.Compose([
            A.HorizontalFlip(p=0.4),
            # A.OneOf([
            #     # A.GridDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            #     # A.ElasticTransform(interpolation=cv2.INTER_NEAREST, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            #     A.ShiftScaleRotate(interpolation=cv2.INTER_NEAREST, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            #     # A.OpticalDistortion(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            # ], p=0.5),
        ], additional_targets={'mask': 'image'})
        afine_transformed = afine_transform(image=img, mask=mask)
        img = afine_transformed["image"]
        mask = afine_transformed["mask"]

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
                A.RandomSnow(p=1.0),
                A.RandomSunFlare(p=1.0)
            ], p=0.05),
        ])
        transformed = transform(image=img)
        img = transformed["image"]

        return img, mask

    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        # Add input_data
        img_t0 = cv2.imdecode(np.frombuffer(raw_data[0]["img"], np.uint8), cv2.IMREAD_COLOR)
        img_t0, _ = resize_img(img_t0, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)
        img_t0 = img_t0.astype(np.float32)

        img_t1 = cv2.imdecode(np.frombuffer(raw_data[1]["img"], np.uint8), cv2.IMREAD_COLOR)
        img_t1, _ = resize_img(img_t1, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)
        img_t1 = img_t1.astype(np.float32)

        # currently hardcoded
        intr = np.array([
            [375.0,  0.0, 160.0],
            [ 0.0, 375.0, 128.0],
            [ 0.0,   0.0,   1.0]
        ], dtype=np.float32)

        input_data = [img_t0, img_t1, intr]

        return raw_data, input_data, None, piped_params
