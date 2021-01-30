import cv2
import numpy as np
from models.centernet import process_2d_output
from common.utils import Roi
from data.od_spec import OD_CLASS_MAPPING


class TestPosProcessing:
    def fill_obj(self, obj, nb_classes, info):
        obj[nb_classes:] = [
            *info["loc_off"],
            info["width"], info["height"],
            *info["bottom_left_off"],
            *info["bottom_right_off"],
            *info["bottom_center_off"],
            info["center_height"],
            info["radial_dist"],
            info["orientation"],
            info["obj_width"],
            info["obj_height"],
            info["obj_length"]
        ]

    def test_post_processing(self):
        nb_classes = 3
        channels = nb_classes + 16
        output_mask = np.zeros((9, 11, channels))

        # Add obj 1
        testObj1 = output_mask[4][5]
        testObj1[0] = 0.8 # set class
        self.fill_obj(testObj1, nb_classes, {
            "width": 7, "height": 6, "loc_off": [0.1, 0.2], "bottom_left_off": [-2, 1], "bottom_right_off": [2, 1], "bottom_center_off": [0, 2],
            "center_height": 3.0, "radial_dist": 30.0, "orientation": 0.0, "obj_width": 1.5, "obj_height": 1.0, "obj_length": 2.0
        })

        r = 2.0
        roi = Roi()
        roi.scale = 0.25
        roi.offset_left = -5
        roi.offset_top = -3
        objects = process_2d_output(output_mask, roi, r, nb_classes)

        input_img = np.zeros((72, 128, 3))
        for obj in objects:
            color = list(OD_CLASS_MAPPING.values())[obj["obj_idx"]]
            top_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1] - obj["center_height"]))
            bottom_left = (int(obj["bottom_left"][0]), int(obj["bottom_left"][1]))
            bottom_center = (int(obj["bottom_center"][0]), int(obj["bottom_center"][1]))
            bottom_right = (int(obj["bottom_right"][0]), int(obj["bottom_right"][1]))
            cv2.line(input_img, bottom_left, bottom_center, (0, 255, 0) , 1) 
            cv2.line(input_img, bottom_center, bottom_right, (0, 255, 0) , 1) 
            cv2.line(input_img, bottom_center, top_center, (0, 255, 0) , 1)

            top_left = (int(obj["fullbox"][0]), int(obj["fullbox"][1]))
            bottom_right = (int(obj["fullbox"][0] + obj["fullbox"][2]), int(obj["fullbox"][1] + obj["fullbox"][3]))
            cv2.rectangle(input_img, top_left, bottom_right, (255, 0, 0), 1)

            cv2.circle(input_img, (int(obj["center"][0]), int(obj["center"][1])), 2, (0, 0, 255), 1)

        cv2.imshow("example_img", input_img)
        cv2.waitKey(0)
