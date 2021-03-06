class Params:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 16
        self.PLANED_EPOCHS = 90
        self.LOAD_WEIGHTS = None

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size

        self.MASK_WIDTH = (self.INPUT_WIDTH // 2) # width of the output mask
        self.MASK_HEIGHT = (self.INPUT_HEIGHT // 2) # height of the output mask
