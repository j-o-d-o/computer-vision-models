"""
Hold all the specific parameters in this python class
"""


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    INPUT_WIDTH = 612
    INPUT_HEIGHT = 185
    INPUT_CHANNELS = 3

    ########################
    ### HYPER PARAMETERS ###
    ########################
    R = 2 # Scale from input image to output heat map
    FOCAL_LOSS_ALPHA = 2
    FOCAL_LOSS_BETA = 2
    LOSS_KEYPOINT_WEIGHT = 1
    LOSS_OFFSET_WEIGHT = 1
    LOSS_BOX_SIZE_WEIGHT = 5

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 4
    BATCH_SIZE = 4

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_PATH = None