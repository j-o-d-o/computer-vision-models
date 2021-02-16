class DmdsParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 6
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None
        self.LOAD_DEPTH_MODEL = None # "/home/computer-vision-models/trained_models/depth_ds_2021-02-15-121653/tf_model_0/keras.h5" # weights just used for the encoder for depth model part

        # Input
        self.INPUT_WIDTH = 320 # width of input img in [px]
        self.INPUT_HEIGHT = 128 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size
