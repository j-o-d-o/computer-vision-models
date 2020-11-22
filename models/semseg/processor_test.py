import cv2
import numpy as np
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger
from models.semseg.processor import ProcessImages
from models.semseg.inference import to_3channel


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

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
            batch_size=3,
            processors=[ProcessImages()]
        )

        batch_x, batch_y = train_gen[0]

        for i, input_data in enumerate(batch_x):
            assert len(input_data) > 0
            mask_img = to_3channel(batch_y[i])
            cv2.imshow("img", input_data.astype(np.uint8))
            cv2.imshow("mask", mask_img)
            cv2.waitKey(0)
