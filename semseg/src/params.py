"""
Hold all the specific parameters in this python class
"""


class Params:
  ########################
  ##### INPUT CONFIG #####
  ########################
  IMG_WIDTH = 320
  IMG_HEIGHT = 130
  IMG_CHANNELS = 3
  OFFSET_TOP = 218 # in pixel of original size, set to None if everything should be cut from top to get to goal aspect ratio

  MASK_WIDTH = IMG_WIDTH
  MASK_HEIGHT = IMG_HEIGHT

  ########################
  ### TRAIN PARAMETERS ###
  ########################
  NUM_EPOCH = 38
  BATCH_SIZE = 16

  ########################
  ### TRAINING FLAGS #####
  ########################
  LOAD_PATH = None
