import cv2
import numpy as np
from models.centernet_2d import process_2d_output
from common.utils import Roi
from data.od_spec import OD_CLASS_MAPPING


class TestPosProcessing:
    def test_post_processing(self):
        output_mask = np.zeros((9, 11, 5))
        output_mask[4][5][0] = 0.8
        output_mask[4][5][3] = 0.7
        output_mask[4][5][4] = 0.4
        output_mask[2][6][1] = 1.0
        output_mask[2][6][3] = 1.0
        output_mask[2][6][4] = 1.0
        r = 2.0
        nb_classes = 3
        roi = Roi()
        roi.scale = 0.25
        roi.offset_left = -20
        objects = process_2d_output(output_mask, roi, r, nb_classes)

        input_img = np.zeros((72, 128, 3))
        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["obj_idx"]]
            cv2.rectangle(input_img, obj["top_left"], obj["bottom_right"], (color[0]/255, color[1]/255, color[2]/255), 1)
            
        cv2.imshow("example_img", input_img)
        cv2.waitKey(0)
