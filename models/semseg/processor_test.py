import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from models.semseg.params import SemsegParams
from models.semseg.processor import ProcessImages
from data.semseg_spec import SEMSEG_CLASS_MAPPING


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = SemsegParams()

        # get one entry from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "semseg", "comma10k")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=30
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=10,
            processors=[ProcessImages(self.params)]
        )

        batch_x, batch_y = train_gen[0]

        for i, input_data in enumerate(batch_x):
            assert len(input_data) > 0
            cls_items = list(SEMSEG_CLASS_MAPPING.items())
            nb_classes = len(cls_items)
            mask_img = to_3channel(batch_y[i], cls_items, threshold=0.999)

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
            plt.show()
