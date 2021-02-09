import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from models.depth.params import DepthParams
from models.depth.processor import ProcessImages


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = DepthParams()

        # get one entry from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "depth", "driving_stereo")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            data_split=(70, 30),
            limit=100,
        )

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=20,
            processors=[ProcessImages(self.params)]
        )

        batch_x, batch_y = train_gen[0]

        for i, input_data in enumerate(batch_x):
            assert len(input_data) > 0
            mask_img = batch_y[i] * 256
            mask_img = mask_img.astype(np.uint16)

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cv2.cvtColor(input_data.astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))
            ax2.imshow(mask_img, cmap='gray', vmin=0, vmax=25000)
            plt.show()
