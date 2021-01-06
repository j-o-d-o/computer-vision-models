"""
Hold all the specific parameters in this python class
"""


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    OFFSET_BOTTOM = -200 # relative to org img size
    INPUT_WIDTH = 320
    INPUT_HEIGHT = 130
    INTPUT_CHANNELS = 3
    MASK_WIDTH = INPUT_WIDTH
    MASK_HEIGHT = INPUT_HEIGHT

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 38
    BATCH_SIZE = 8

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_MODEL_PATH = None
