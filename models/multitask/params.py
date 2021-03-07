from models.centernet.params import CenternetParams
from models.semseg.params import SemsegParams
from models.depth.params import Params as DepthParams
from data.label_spec import SEMSEG_CLASS_MAPPING


class MultitaskParams():
    def __init__(self, nb_classes):
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 256
        self.MASK_WIDTH = (self.INPUT_WIDTH // 2) # width of the output mask
        self.MASK_HEIGHT = (self.INPUT_HEIGHT // 2) # height of the output mask

        self.PLANED_EPOCHS = 100
        self.BATCH_SIZE = 4
        self.LOAD_WEIGHTS = "/home/computer-vision-models/depth_keras.h5"

        self.cn_params = CenternetParams(nb_classes)
        self.cn_params.LOAD_WEIGHTS = "/home/computer-vision-models/centernet_keras.h5"
        self.cn_params.CHANNELS = 3 + len(SEMSEG_CLASS_MAPPING.items()) + 1
        self.cn_params.REGRESSION_FIELDS["class"].active = True
        self.cn_params.REGRESSION_FIELDS["r_offset"].active = True
        self.cn_params.REGRESSION_FIELDS["fullbox"].active = True
        self.cn_params.REGRESSION_FIELDS["l_shape"].active = False
        self.cn_params.REGRESSION_FIELDS["3d_info"].active = False
        self.cn_params.REGRESSION_FIELDS["track_offset"].active = False

        self.semseg_params = SemsegParams()
        self.semseg_params.LOAD_WEIGHTS = "/home/computer-vision-models/semseg_keras.h5"

        self.depth_params = DepthParams()
        self.depth_params.LOAD_WEIGHTS = "/home/computer-vision-models/depth_keras.h5"
