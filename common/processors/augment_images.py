import cv2
from common.processors import IPreProcessor
from random import seed
from random import random


class AugmentImages(IPreProcessor):
    def process(self, raw_data, input_data, ground_truth, piped_params=None):
        alpha_unscaled = random()
        min = 0.4
        max = 1.7
        alpha_scaled = min + (alpha_unscaled * (max - min))

        beta_unscaled = random()
        min = 0.4
        max = 1.7
        beta_scaled = min + (beta_unscaled * (max - min))
        input_data = cv2.convertScaleAbs(input_data, alpha=alpha_scaled, beta=beta_scaled)

        return raw_data, input_data, ground_truth, piped_params
