"""
Wrapper for all specific parameters needed to train the centernet
"""
from collections import OrderedDict
from dataclasses import dataclass
import re


class Params:
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
        self.LOAD_PATH = None
        
        # Input
        self.INPUT_WIDTH = 608 # width of input img in [px]
        self.INPUT_HEIGHT = 144 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size
        
        # Output Mask
        self.R = 2 # scale from input image to output heat map
        self.VARIANCE_ALPHA = 0.90 # variance to determine how spread out the class blobs are on the ground truth

        # Loss - Class
        self.FOCAL_LOSS_ALPHA = 2.0
        self.FOCAL_LOSS_BETA = 4.0
        self.CLASS_WEIGHT = 1.0
        self.NB_CLASSES = nb_classes # need to calc the indices of all the regression fields
        # Loss - Regression
        self.REGRESSION_FIELDS = OrderedDict([
            ("r_offset", Params.RegressionField(True, 2, 0.7, "x, y")),
            ("fullbox", Params.RegressionField(False, 2, 0.1, "width, height (in [px] relative to input)")),
            ("l_shape", Params.RegressionField(False, 7, 0.1, "bottom_left_offset, bottom_right_offset, bottom_center_offset, center_height, (all points (x,y) in [px] relative to input)")),
            ("3d_info", Params.RegressionField(False, 5, [0.1, 0.2, 0.1], "radial_dist [m], orientation [rad], width, height, length [m] (all in cam coordinate system)")),
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
            "mask": [self.INPUT_HEIGHT // self.R, self.INPUT_WIDTH // self.R, self.mask_channels()],
            "output_fields": []
        }
        dict_data["output_fields"].append({"object_class": {"start_idx": 0, "end_idx": self.NB_CLASSES - 1, "comment": "Object classes"}})
        for key, field in self.REGRESSION_FIELDS.items():
            if field.active:
                dict_data["output_fields"].append({key: {"start_idx": self.start_idx(key), "end_idx": self.end_idx(key), "comment": field.comment}})
        return dict_data

    def fill_from_json(self):
        #TODO: Fill form a json file that contains the serialized params
        pass
