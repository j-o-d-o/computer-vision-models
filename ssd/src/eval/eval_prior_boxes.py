from mlpipe.utils import Config
from mlpipe.data_reader.mongodb import MongoDBConnect
from ssd.src.params import Params
from ssd.src.prior_boxes import PriorBoxes
from ssd.src.processor import ProcessImage, GenGroundTruth
import matplotlib.pyplot as plt


if __name__ == "__main__":
  # Get all the training data the eval should be done with
  Config.add_config('./config.ini')
  collection_details = ("localhost_mongo_db", "object_detection", "kitty_training")

  mongo_con = MongoDBConnect()
  mongo_con.add_connections_from_config(Config.get_config_parser())
  collection = mongo_con.get_collection("localhost_mongo_db", "object_detection", "kitty_training")

  db_cursor = collection.find({}).limit(100)
  gt_boxes = []
  gt_classes = []
  img_processor = ProcessImage()
  prior_boxes = PriorBoxes(clip_boxes=True)
  counts = []
  ious = []

  i = 0

  for entry in db_cursor:
    i += 1
    print(i)
    # apply image processor to get roi
    input_data = None
    gt_data = None
    piped_params = {}
    raw_data = entry
    raw_data, input_data, gt_data, piped_params = img_processor.process(raw_data, input_data, gt_data, piped_params)

    for obj in entry["objects"]:
      # obj["bbox"] is stored with [tx, ty, width, height] with [tx, ty] being the top left corner
      top_x = obj["bbox"][0]
      top_y = obj["bbox"][1]
      width = obj["bbox"][2]
      height = obj["bbox"][3]
      bbox = [top_x + width * 0.5, top_y + height * 0.5, width, height]

      bbox, ratio = GenGroundTruth._apply_roi(bbox, piped_params["roi"])
      pixels = bbox[2] * bbox[3]
      bbox = GenGroundTruth._normalize_box(bbox)

      if obj["obj_class"] in Params.CLASSES and 0.1 < ratio < 10 and pixels > 84:
        gt_classes.append(Params.CLASSES.index(obj["obj_class"]))
        gt_boxes.append(bbox)
      else:
        print("Ups, Ratio bad or Class does not exist or too small")
        # raise ValueError("Class: " + str(obj["obj_class"]) + " does not exist")

      # Match to prior boxes and check match scores
      _, iou_map = prior_boxes.match(gt_boxes, gt_classes, iou_threshold=0.4)

      for e in iou_map:
        count = len(e["iou"])
        counts.append(count)
        ious.append(sum(e["iou"]) / count)

  avg_counts = sum(counts) / len(counts)
  avg_ious = sum(ious) / len(ious)
  print(avg_counts)
  print(avg_ious)

  # plot lists
  fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

  # We can set the number of bins with the `bins` kwarg
  axs[0].hist(counts, bins=100)
  axs[1].hist(ious, bins=100)

  plt.show()
