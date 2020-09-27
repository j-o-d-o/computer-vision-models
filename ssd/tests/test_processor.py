from typing import List
from ssd.src.params import Params
from ssd.src.prior_boxes import PriorBoxes, PriorBoxDecoder
from ssd.src.processor import ProcessImage, GenGroundTruth, AugmentImg
from mlpipe.data_reader.mongodb import load_ids, MongoDBGenerator
from mlpipe.utils import Config, MLPipeLogger
import cv2


class TestProcessors:
  def setup_method(self):
    """
    Set up parameters for test methods
    """
    MLPipeLogger.init()
    MLPipeLogger.remove_file_logger()

    # get one entry from the database
    Config.add_config('../../config.ini')
    self.collection_details = ("localhost_mongo_db", "object_detection", "kitty_training")

    # Create Data Generators
    self.train_data, self.val_data = load_ids(
      self.collection_details,
      data_split=(70, 30),
      limit=30
    )

  @staticmethod
  def show_img(img, boxes: List[List[float]] = None):
    if boxes is not None:
      for box in boxes:
        cx, cy, w, h = box[0] * Params.IMG_WIDTH
        cv2.rectangle(img, (int(cx - w/2), int(cy + h/2)), (int(cx + w/2), int(cy - h/2)), (0, 255, 0), 1)
    cv2.imshow("debug test img", img)
    cv2.waitKey(0)

  def test_process_image(self):
    train_gen = MongoDBGenerator(
      self.collection_details,
      self.train_data,
      batch_size=3,
      processors=[ProcessImage()]
    )

    batch_x, batch_y = train_gen[0]

    for input_data in batch_x:
      assert len(input_data) > 0
      # TestProcessors.show_img(input_data)

  def test_gen_ground_truth(self):
    pb = PriorBoxes(clip_boxes=True)
    # pb.save_to_file("test.json")
    pb_decoder = PriorBoxDecoder(pb.get_flattened_boxes())
    # pb_decoder.init_boxes_from_file("test.json")
    train_gen = MongoDBGenerator(
      self.collection_details,
      self.train_data,
      batch_size=3,
      processors=[ProcessImage(), AugmentImg(), GenGroundTruth(pb)]
    )

    for train_data in train_gen:
      batch_x, batch_y = train_data

      for input_data, output in zip(batch_x, batch_y):
        # set threshold to 0.0 in case you want to see also all prior boxes (background boxes)
        boxes = pb_decoder.decode_boxes(output, len(Params.CLASSES), threshold=0.5)
        TestProcessors.show_img(input_data, boxes)
