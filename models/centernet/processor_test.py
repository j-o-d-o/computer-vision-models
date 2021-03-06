import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from data.label_spec import OD_CLASS_MAPPING
from numba.typed import List
from models.centernet.processor import ProcessImages
from models.centernet.params import CenternetParams


class TestProcessors:
    def setup_method(self):
        Logger.init()
        Logger.remove_file_logger()

        self.params = CenternetParams(len(OD_CLASS_MAPPING))
        self.params.REGRESSION_FIELDS["l_shape"].active = True
        self.params.REGRESSION_FIELDS["3d_info"].active = True

        # get some entries from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "labels", "nuscenes_train")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=250
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            [self.collection_details],
            [self.train_data],
            batch_size=30,
            processors=[ProcessImages(self.params, start_augmentation=[0, 0], show_debug_img=False)]
        )

        for batch_x, batch_y in train_gen:
            print("New batch")
            for i in range(len(batch_x[0])):
                assert len(batch_x[0]) > 0
                img1 = batch_x[0][i]
                heatmap = np.array(batch_y[i][:, :, :-1]) # needed because otherwise numba makes mimimi
                heatmap = to_3channel(heatmap, List(OD_CLASS_MAPPING.items()), 0.01, True, False)
                weights = np.stack([batch_y[i][:, :, -1]]*3, axis=-1)

                f, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(cv2.cvtColor(batch_x[0][i].astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax2.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                ax3.imshow(weights)
                plt.show()
