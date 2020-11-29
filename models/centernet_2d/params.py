"""
Hold all the specific parameters in this python class
"""


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    INPUT_WIDTH = 308
    INPUT_HEIGHT = 92

    ########################
    ### HYPER PARAMETERS ###
    ########################
    R = 2 # Scale from input image to output heat map
    FOCAL_LOSS_ALPHA = 2
    FOCAL_LOSS_BETA = 4
    LOSS_SIZE_WEIGHT = 0.4
    VARIANCE_ALPHA = 0.90

    #######################
    ### FILTERED LABELS ###
    #######################
    MIN_BOX_WIDTH = 35
    MIN_BOX_HEIGHT = 35

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 20
    BATCH_SIZE = 3

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_PATH = None
