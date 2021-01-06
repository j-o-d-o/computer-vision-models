"""
Hold all the specific parameters in this python class
"""


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    INPUT_WIDTH = 308
    INPUT_HEIGHT = 92
    OFFSET_BOTTOM = 0

    ########################
    ### HYPER PARAMETERS ###
    ########################
    R = 2 # Scale from input image to output heat map
    FOCAL_LOSS_ALPHA = 2
    FOCAL_LOSS_BETA = 4
    LOSS_SIZE_WEIGHT = 0.4
    LOSS_OFFSET_WEIGHT = 0.1
    VARIANCE_ALPHA = 0.90
    BOTTELNECK_ALPHA = 1

    #######################
    ### FILTERED LABELS ###
    #######################
    MIN_BOX_WIDTH = 10 # in pixel relative to INPUT size
    MIN_BOX_HEIGHT = 10 # in pixel relative to INPUT size

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 50
    BATCH_SIZE = 8

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_PATH = None
