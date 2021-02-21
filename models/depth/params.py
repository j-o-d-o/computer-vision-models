class Params:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 12
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None #"/home/computer-vision-models/trained_models/depth_ds_2021-02-18-181020/tf_model_0/keras.h5"

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size
