"""
Hold all the specific parameters in this python class
"""

class SemsegParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 2
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None

        # Input
        self.INPUT_WIDTH = 320 # width of input img in [px]
        self.INPUT_HEIGHT = 92 # height of input img in [px]
        self.OFFSET_BOTTOM = 100 # offset in [px], applied before scaling, thus relative to org. img s
        self.MASK_WIDTH = 320 # width of the output mask
        self.MASK_HEIGHT = 92 # height of the output mask
