"""
Hold all the specific parameters in this python class
"""


class Params:
    ########################
    ##### INPUT CONFIG #####
    ########################
    OFFSET_BOTTOM = -260 # relative to org img size
    INPUT_WIDTH = 320
    INPUT_HEIGHT = 92
    INTPUT_CHANNELS = 3
    MASK_WIDTH = 320
    MASK_HEIGHT = 92

    ########################
    ### TRAIN PARAMETERS ###
    ########################
    NUM_EPOCH = 38
    BATCH_SIZE = 8

    ########################
    ### TRAINING FLAGS #####
    ########################
    LOAD_MODEL_PATH = "/home/jo/git/computer-vision-models/trained_models/semseg_2021-01-09-152324/tf_model_8"
