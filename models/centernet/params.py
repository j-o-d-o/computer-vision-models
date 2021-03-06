"""
Wrapper for all specific parameters needed to train the centernet
"""
from collections import OrderedDict
from dataclasses import dataclass
import re
import json


class CenternetParams:
    @dataclass
    class RegressionField:
        active: bool = False # if the field is active it will be added to the output
        size: int = 0 # how many float values does it contain
        loss_weight: [float, [float]] = 0.0 # weight that is used for the loss function
        comment: str = "" # describe how the float values are ordered

    def __init__(self, nb_classes: int):
        # Training
        self.BATCH_SIZE = 8
        self.PLANED_EPOCHS = 90
        self.LOAD_WEIGHTS = None

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size

        # Box filters
        self.MIN_BOX_AREA = 15.0 # In [px] relative to INPUT_WIDTH and INPUT_HEIGHT

        # Output Mask
        self.R = 2 # scale from input image to output heat map
        self.VARIANCE_ALPHA = 0.9 # variance to determine how spread out the class blobs are on the ground truth
        self.MASK_HEIGHT = self.INPUT_HEIGHT // self.R
        self.MASK_WIDTH = self.INPUT_WIDTH // self.R

        # Loss - Class
        self.FOCAL_LOSS_ALPHA = 2.0
        self.FOCAL_LOSS_BETA = 4.0
        self.CLASS_WEIGHT = 1.0
        self.NB_CLASSES = nb_classes # need to calc the indices of all the regression fields
        # Loss - Regression
        self.REGRESSION_FIELDS = OrderedDict([
            ("r_offset", CenternetParams.RegressionField(True, 2, 0.2, "x, y")),
            ("fullbox", CenternetParams.RegressionField(False, 2, 0.1, "width, height (in [px] relative to input)")),
            ("l_shape", CenternetParams.RegressionField(False, 7, 0.1, "bottom_left_offset, bottom_center_offset, bottom_right_offset, center_height, (all points (x,y) in [px] relative to input)")),
            ("3d_info", CenternetParams.RegressionField(False, 5, [0.1, 0.2, 0.1], "radial_dist [m], orientation [rad], width, height, length [m] (all in cam coordinate system)")),
            ("track_offset", CenternetParams.RegressionField(False, 2, 0.1, "x and y offset to track at t-1 relative to input size"))
        ])

    def start_idx(self, regression_key: str) -> int:
        """ Calc start index of a certain key. TODO: If havily used, cache or precalc the results here """
        start_idx = self.NB_CLASSES
        found = False
        for key, field in self.REGRESSION_FIELDS.items():
            if field.active:
                if key == regression_key:
                    found = True
                    break
                start_idx += field.size
        assert(found and f"Key: {regression_key} is not active or does not exist!")
        return start_idx

    def end_idx(self, regression_key: str) -> int:
        field = self.REGRESSION_FIELDS[regression_key]
        end_idx = self.start_idx(regression_key) + field.size
        return end_idx

    def mask_channels(self):
        mask_size = self.NB_CLASSES
        for key, field in self.REGRESSION_FIELDS.items():
            if field.active:
                mask_size = self.end_idx(key)
        return mask_size

    def serialize(self):
        dict_data = {
            "input": [self.INPUT_HEIGHT, self.INPUT_WIDTH, 3],
            "mask": [self.MASK_HEIGHT, self.MASK_WIDTH, self.mask_channels()],
            "batch_size": self.BATCH_SIZE,
            "load_weights": self.LOAD_WEIGHTS,
            "output_fields": []
        }
        dict_data["output_fields"].append({"object_class": {"start_idx": 0, "end_idx": self.NB_CLASSES - 1, "comment": "Object classes"}})
        for key, field in self.REGRESSION_FIELDS.items():
            if field.active:
                dict_data["output_fields"].append({key: {"start_idx": self.start_idx(key), "end_idx": self.end_idx(key), "comment": field.comment}})
        return dict_data

    def save_to_storage(self, storage_path: str):
        with open(storage_path + "/parameters.json", "w") as outfile:
            json.dump(self.serialize(), outfile)

    def fill_from_json(self):
        #TODO: Fill form a json file that contains the serialized params
        pass
