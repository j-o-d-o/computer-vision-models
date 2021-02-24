import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel, cmap_depth
from models.multitask import MultitaskParams, ProcessImages
from data.label_spec import SEMSEG_CLASS_MAPPING
from numba.typed import List


class TestProcessors:
    def setup_method(self):
        Logger.init()
        Logger.remove_file_logger()

        self.params = MultitaskParams()

        # get one entry from the database
        Config.add_config('./config.ini')
        self.collection_details_semseg = ("local_mongodb", "labels", "comma10k")
        self.collection_details_depth = ("local_mongodb", "labels", "driving_stereo")

        # Create Data Generators
        self.td_semseg, self.val_semseg = load_ids(
            self.collection_details_semseg,
            data_split=(70, 30),
            limit=30
        )
        self.td_depth, self.val_depth = load_ids(
            self.collection_details_depth,
            data_split=(70, 30),
            shuffle_data=True,
            limit=30
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            [self.collection_details_depth, self.collection_details_semseg],
            [self.td_depth, self.td_semseg],
            batch_size=10,
            processors=[ProcessImages(self.params)],
            shuffle_data=True
        )

        batch_x, batch_y = train_gen[0]
        cls_items = List(SEMSEG_CLASS_MAPPING.items())
        nb_classes = len(cls_items)

        for i, input_data in enumerate(batch_x):
            assert len(input_data) > 0
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_BGR2RGB))

            if batch_y[1][i]:
                semseg_img = to_3channel(batch_y[0][i], cls_items, threshold=0.999)
                ax2.imshow(cv2.cvtColor(semseg_img, cv2.COLOR_BGR2RGB))
            elif batch_y[3][i]:
                ax2.imshow(cv2.cvtColor(cmap_depth(batch_y[2][i], vmin=0.1, vmax=255.0), cv2.COLOR_BGR2RGB))

            ax3.imshow(batch_y[2][i], cmap="gray", vmin=0.0, vmax=1.0)
            plt.show()
