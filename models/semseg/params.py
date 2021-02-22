"""
Hold all the specific parameters in this python class
"""

class SemsegParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 6
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None
        self.LOAD_WEIGHTS = "/home/computer-vision-models/trained_models/semseg_comma10k_augment_2021-02-22-204129/tf_model_3/keras.h5"

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img s
        self.MASK_WIDTH = self.INPUT_WIDTH # width of the output mask
        self.MASK_HEIGHT = self.INPUT_HEIGHT # height of the output mask
