"""
Hold all the specific parameters in this python class
"""

class MultitaskParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 6
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None
        self.LOAD_WEIGHTS_SEMSEG = None # "/home/computer-vision-models/trained_models/semseg_comma10k_augment_2021-02-23-134354/tf_model_44/keras.h5"
        self.LOAD_WEIGHTS_DEPTH = None # ""

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img s
