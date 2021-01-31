import cv2
import numpy as np
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger, to_3channel
from data.od_spec import OD_CLASS_MAPPING
from models.centernet.processor import ProcessImages
from models.centernet.params import CenternetParams


class TestProcessors:
    def setup_method(self):
        Logger.init()
        Logger.remove_file_logger()

        self.params = CenternetParams(len(OD_CLASS_MAPPING))

        # get some entries from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "object_detection", "kitti")

        # Create Data Generators
        self.train_data, self.val_data = load_ids(
            self.collection_details,
            mongodb_filter={"has_3D_info": True},
            data_split=(70, 30),
            limit=200
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
            mask_img = to_3channel(batch_y[i], OD_CLASS_MAPPING, 0.01, True)
            cv2.imshow("img", input_data.astype(np.uint8))
            cv2.imshow("mask", mask_img)
            cv2.waitKey(0)
