import cv2
import numpy as np
import matplotlib.pyplot as plt
from common.data_reader.mongodb import load_ids, MongoDBGenerator
from common.utils import Config, Logger
from models.dmds.params import DmdsParams
from models.dmds.processor import ProcessImages
from models.dmds.train import adapt_doc_ids


class TestProcessors:
    def setup_method(self):
        """
        Set up parameters for test methods
        """
        Logger.init()
        Logger.remove_file_logger()

        self.params = DmdsParams()

        # get one entry from the database
        Config.add_config('./config.ini')
        collection_details = ("local_mongodb", "depth", "driving_stereo")
        scenes = [
            "2018-10-26-15-24-18",
            "2018-10-19-09-30-39",
        ]
        self.train_data = []
        self.val_data = []
        self.collection_details = []

        # get ids
        for scene_token in scenes:
            train_data, val_data = load_ids(
                collection_details,
                data_split=(70, 30),
                limit=100,
                shuffle_data=False,
                mongodb_filter={"scene_token": scene_token},
                sort_by={"timestamp": 1}
            )
            train_data = adapt_doc_ids(train_data)
            val_data = adapt_doc_ids(val_data)
            self.train_data.append(train_data)
            self.val_data.append(val_data)
            self.collection_details.append(collection_details)

    def test_process_image(self):
        train_gen = MongoDBGenerator(
            self.collection_details,
            self.train_data,
            batch_size=7,
            processors=[ProcessImages(self.params)],
            data_group_size=2,
            continues_data_selection=False,
            shuffle_data=False
        )

        for batch_x, _ in train_gen:
            for i in range(len(batch_x[0])):
                assert len(batch_x[0]) > 0
                img_t0 = batch_x[0][i]
                img_t1 = batch_x[1][i]

                f, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(cv2.cvtColor(img_t0.astype(np.uint8), cv2.COLOR_BGR2RGB))
                ax2.imshow(cv2.cvtColor(img_t1.astype(np.uint8), cv2.COLOR_BGR2RGB))
                #plt.show()
                plt.draw()
                plt.waitforbuttonpress(0) # this will wait for indefinite time
                plt.close(f)
