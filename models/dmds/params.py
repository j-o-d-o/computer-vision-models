class DmdsParams:
    def __init__(self):
        # Training
        self.BATCH_SIZE = 2
        self.PLANED_EPOCHS = 90
        self.LOAD_PATH = None
        self.LOAD_DEPTH_MODEL = "/home/computer-vision-models/trained_models/depth_ds_2021-02-19-123248/tf_model_0/keras.h5"

        # Input
        self.INPUT_WIDTH = 640 # width of input img in [px]
        self.INPUT_HEIGHT = 256 # height of input img in [px]
        self.OFFSET_BOTTOM = 0 # offset in [px], applied before scaling, thus relative to org. img size

        # Loss weights
        self.rgb_cons = (1.0 / 255.0)
        self.ssim_cons = 3.0
        self.depth_cons = 0.0
        self.supervise_depth = 0.5
        self.depth_smoothing = 0.001
        self.var_depth = 1e-6
        self.rot_cyc = 1.0e-3
        self.tran_cyc = 1.0e-2
        self.mot_smoothing = 1e-4
        self.mot_drift = 1e-6
