import numpy as np
import cv2
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from common.processors import IPreProcessor
from common.utils import resize_img
from models.depth.params import DepthParams
import albumentations as A
import matplotlib.pyplot as plt


class ProcessImages(IPreProcessor):
    def __init__(self, params: DepthParams):
        self.params: DepthParams = params

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
        img_encoded = np.frombuffer(raw_data["img"], np.uint8)
        input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
        input_data, roi_img = resize_img(input_data, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM)

        # Add ground_truth mask
        mask_encoded = np.frombuffer(raw_data["depth"], np.uint8)
        mask_img = cv2.imdecode(mask_encoded, cv2.IMREAD_ANYDEPTH)
        mask_img, _ = resize_img(mask_img, self.params.INPUT_WIDTH, self.params.INPUT_HEIGHT, offset_bottom=self.params.OFFSET_BOTTOM, interpolation=cv2.INTER_NEAREST)

        # augment and resize mask to real size
        input_data, mask_img = self.augment(input_data, mask_img)
        ground_truth, _ = resize_img(mask_img, self.params.MASK_WIDTH, self.params.MASK_HEIGHT, offset_bottom=0, interpolation=cv2.INTER_LINEAR)

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(mask_img, cmap='gray', vmin=0, vmax=25000)
        ax2.imshow(ground_truth, cmap='gray', vmin=0, vmax=25000)
        plt.show()

        input_data = input_data.astype(np.float32)
        ground_truth = ground_truth.astype(np.float32) / 256.0
        ground_truth = ground_truth.reshape((ground_truth.shape[0], ground_truth.shape[1], 1))
        return raw_data, input_data, ground_truth, piped_params
