import cv2
import numpy as np
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger
from data.od_spec import OD_CLASS_MAPPING
from models.centernet_2d.processor import ProcessImages


def to_3channel(raw_mask_output, class_mapping):
    array = np.zeros((raw_mask_output.shape[0] * raw_mask_output.shape[1] * 3), dtype='uint8')
    flattened_arr = raw_mask_output.reshape((-1, raw_mask_output.shape[2]))
    for i, one_hot_encoded_arr in enumerate(flattened_arr):
        # find index of highest value in the one_hot_encoded_arr
        nb_classes = len(class_mapping)
        if np.max(one_hot_encoded_arr[:nb_classes]) > 0.05:
            cls_idx = np.argmax(one_hot_encoded_arr[:nb_classes])
            # convert index to hex value
            cls_colour = list(class_mapping.items())[int(round(cls_idx))][1]
            # fill new array with BGR values
            new_i = i * 3
            array[new_i] = cls_colour[0]
            array[new_i + 1] = cls_colour[1]
            array[new_i + 2] = cls_colour[2]

    return array.reshape((raw_mask_output.shape[0], raw_mask_output.shape[1], 3))


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        # get some entries from the database
        Config.add_config('./config.ini')
        self.collection_details = ("local_mongodb", "object_detection", "kitti")

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
            mask_img = to_3channel(batch_y[i], OD_CLASS_MAPPING)
            cv2.imshow("img", input_data.astype(np.uint8))
            cv2.imshow("mask", mask_img)
            cv2.waitKey(0)
