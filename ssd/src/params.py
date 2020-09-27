"""
Hold all the specific parameters in this python class
"""
from typing import NamedTuple, List, Tuple


class FeatureMapConfig(NamedTuple):
  """
  # size: size [x,y] of the feature map in pixels
  # ratios: ratio of the boxes [x,y].
  #         e.g. [1.4, 0.6] would stretch the width to 1.4 of the original scale and reduce the height to 0.6.
  """
  size: Tuple[int, int]
  ratios: List[Tuple[float, float]]


class Params:
  ########################
  ##### INPUT CONFIG #####
  ########################
  IMG_WIDTH = 512
  IMG_HEIGHT = 192
  IMG_CHANNELS = 3

  # These should be the same as in the label spec
  CLASSES = ["BACKGROUND", "CAR", "TRUCK", "VAN", "MOTORBIKE", "BICYCLE", "PED"]

  ########################
  ### PRIOR BOX CONFIG ###
  ########################
  # must be largest to smallest, if edit the sizes or number of feature maps, also edit the model (in model.py)
  # TODO: Adjust scales to fit the wide image (scales are relative to width). E.g. the negative scales h > w should
  #       be much smaller scaled.
  FEATURE_MAPS = [
    # FeatureMapConfig(size=(16, 64), ratios=[(2.4, 2.4), (2.3, 2.8), (2.8, 2.3)]),
    FeatureMapConfig(size=(12, 32), ratios=[(3.0, 3.0), (2.0, 2.0), (1.5, 1.5),
                                            (1.8, 3.2), (2.3, 2.8),
                                            (3.2, 1.8), (2.8, 2.3)]),
    FeatureMapConfig(size=( 6, 16), ratios=[(3.0, 3.0), (2.0, 2.0), (1.5, 1.5),
                                            (1.8, 3.2), (2.3, 2.8),
                                            (3.2, 1.8), (2.8, 2.3)]),
    FeatureMapConfig(size=( 3,  8), ratios=[(3.0, 3.0), (2.0, 2.0), (1.5, 1.5),
                                            (1.8, 3.2), (2.3, 2.8),
                                            (3.2, 1.8), (2.8, 2.3)]),
  ]

  SCALE_MIN = 0.02
  SCALE_MAX = 0.25

  ########################
  ### HYPER PARAMETERS ###
  ########################
  NEG_POS_RATIO = 3  # e.g. 3 -> 3:1 negative:positive ratio

  ########################
  ### TRAIN PARAMETERS ###
  ########################
  NUM_EPOCH = 22
  BATCH_SIZE = 4
  ALPHA = 1

  ########################
  ### TRAINING FLAGS #####
  ########################
  LOAD_PATH = None # "/home/jodo/trained_models/kitti_mobile_ssd_17-12-2019-09-33-35/keras_model_5.h5"
