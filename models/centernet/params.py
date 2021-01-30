"""
Hold all the specific parameters in this python class
"""
from collections import OrderedDict


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    INPUT_WIDTH = 608
    INPUT_HEIGHT = 144
    OFFSET_BOTTOM = 0

    ########################
    ### HYPER PARAMETERS ###
    ########################
    R = 2 # Scale from input image to output heat map
    VARIANCE_ALPHA = 0.90
    BOTTELNECK_ALPHA = 1
    # loss params
    FOCAL_LOSS_ALPHA = 2.0
    FOCAL_LOSS_BETA = 4.0
    CLS_WEIGHT = 1.0
    OFFSET_WEIGHT = 0.5
    SIZE_WEIGHT = 0.0 #0.1
    BOX3D_WEIGHT = 0.0 #0.1
    RADIAL_DIST_WEIGHT = 0.0 # 0.5 # is using mape
    ORIENTATION_WEIGHT = 0.0 # 0.15 # is using some special thingy
    OBJ_DIMS_WEIGHT = 0.0 # 0.15

    #######################
    ### FILTERED LABELS ###
    #######################
    MIN_BOX_WIDTH = 4 # in pixel relative to INPUT size
    MIN_BOX_HEIGHT = 4 # in pixel relative to INPUT size

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 120
    BATCH_SIZE = 4

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_PATH = None
