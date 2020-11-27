import cv2
import pytest
import numpy as np
from common.utils import Logger, resize_img


class TestImage:
    @pytest.mark.parametrize("org_width, org_height, goal_width, goal_height, offset_bottom", [
        (1200, 1200, 480, 320, 0),
        (1200, 1200, 480, 320, -60),
        (1200, 1200, 480, 320, 60),
        (1200, 320, 480, 320, 0)
    ])
    def test_resize_img(self, org_width, org_height, goal_width, goal_height, offset_bottom):
        img = np.zeros((org_height, org_width))
        resized_img, roi = resize_img(img, goal_width, goal_height, offset_bottom)
        assert resized_img.shape[0] == goal_height
        assert resized_img.shape[1] == goal_width

    