import numpy as np
import cv2
import random
from typing import List
from dataclasses import dataclass
from ssd.src.prior_boxes import PriorBoxes
from ssd.src.params import Params
from mlpipe.processors.i_processor import IPreProcessor


@dataclass
class Roi:
  # offsets are pre scaling!
  offset_top: int = 0
  offset_bottom: int = 0
  offset_left: int = 0
  offset_right: int = 0
  scale: float = 1.0


class ProcessImage(IPreProcessor):
  def __init__(self, dataset: str = None):
    """
    Create image data from database and resize to Params.IMG_WIDTH, Params.IMG_HEIGHT
    adds ["roi"] field to piped_params
    :param dataset: Type of dataset e.g. "KITTI", some datasets might need specific rois (e.g. cut away hood of car
    """
    self._dataset: str = dataset

  @staticmethod
  def resize_img(img: np.ndarray) -> (np.ndarray, Roi):
    """
    Resize image in a way that it fits the params, the default cropping will take delta height from top and delta width
    from left and right border equally
    :param img: numpy img array (as used by cv2), note that it will also be changed in place
    :return: scaled and cropped image, roi data
    @pre: Depends on Params.IMG_WIDTH and Params.IMG_HEIGHT
    """
    # offsetting is done prior to scaling!
    roi_data = Roi()

    # In case a dataset needs specific offsets, add here...
    # h, w = img.shape[:2]
    # img = img[:(h-50), :]

    h, w = img.shape[:2]
    curr_ratio = w / h
    target_ratio = Params.IMG_WIDTH / Params.IMG_HEIGHT
    if curr_ratio > target_ratio:
      # cut width equally from left and right edge
      delta_width = int((w - (target_ratio * h)) / 2)
      img = img[:, delta_width:(w-delta_width)]
      roi_data.offset_left += delta_width
      roi_data.offset_right += delta_width
    else:
      # cut of height from the top of the image
      delta_height = int(h - (w / target_ratio))
      img = img[delta_height:]
      roi_data.offset_top += delta_height
    unscaled_h, unscaled_w = img.shape[:2]
    roi_data.scale = Params.IMG_WIDTH / unscaled_w
    img = cv2.resize(img, (Params.IMG_WIDTH, Params.IMG_HEIGHT))
    return img, roi_data

  def process(self, raw_data, input_data, ground_truth, piped_params=None):
    img_encoded = np.frombuffer(raw_data["img"], np.uint8)
    input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    input_data, roi = self.resize_img(input_data)
    piped_params["roi"] = roi
    return raw_data, input_data, ground_truth, piped_params


class AugmentImg(IPreProcessor):
  @staticmethod
  def _rnd_brightness(img: np.ndarray):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    inc = random.randint(-100, 100)
    v = v.astype(np.int)
    v += inc
    v = np.clip(v, 0, 255)
    v = v.astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

  @staticmethod
  def _rnd_blur(img):
    k_size = random.randint(0, 1)
    if k_size == 0:
      k_size = (3, 3)
    else:
      k_size = (5, 5)
    sigma_x = random.uniform(2.5, 25)
    img = cv2.GaussianBlur(img, k_size, sigma_x)
    return img

  @staticmethod
  def _rnd_sharp(img):
    kernel = np.array([ [ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel)
    return img

  def process(self, raw_data, input_data, ground_truth, piped_params=None):
    random_augmentation = random.randrange(5)
    if random_augmentation == 1:
      input_data = self._rnd_brightness(input_data)
    elif random_augmentation == 2:
      input_data = self._rnd_blur(input_data)
    elif random_augmentation == 3:
      input_data = self._rnd_sharp(input_data)
    return raw_data, input_data, ground_truth, piped_params


class GenGroundTruth(IPreProcessor):
  def __init__(self, prior_boxes: PriorBoxes):
    """
    Create ground truth info which is used to check against the output during training
    :param prior_boxes: A initialized instance of PriorBoxes
    """
    self._prior_boxes = prior_boxes

  @staticmethod
  def _debug_show_img(img: np.ndarray, boxes: List[List[float]]):
    for box in boxes:
      cx, cy, w, h = box
      cx *= Params.IMG_WIDTH
      w *= Params.IMG_WIDTH
      cy *= Params.IMG_WIDTH
      h *= Params.IMG_WIDTH
      cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 1)
    cv2.imshow("debug test img (gt boxes)", img)
    cv2.waitKey(0)

  @staticmethod
  def _apply_roi(box: List[float], roi: Roi) -> [List[float], float]:
    """
    Apply roi from processing the image to the bounding box
    :param box: [cx, cy, width, height] in unscaled image pixel coordinates (not normalized!)
    :param roi: ROI info
    :return: cropped and scaled [cx, cy, width, height]
    """
    new_rect = [
      (box[0] - roi.offset_left) * roi.scale,
      (box[1] - roi.offset_top) * roi.scale,
      box[2] * roi.scale,
      box[3] * roi.scale
    ]
    ratio = 0
    if new_rect[3] > 0:
      ratio = new_rect[2] / new_rect[3]
    return new_rect, ratio

  @staticmethod
  def _normalize_box(box: List[float]) -> List[float]:
    """
    Normalize box coordinates with IMAGE_WIDTH
    :param box: List of floats [cx, cy, width, height] in unscaled image coordinates
    :return: List floats [cx, cy, width, height] in normalized (to IMAGE_WIDTH) image coordinates
    @pre: Depends on Params.IMG_WIDTH
    """
    return [box[0] / Params.IMG_WIDTH, box[1] / Params.IMG_WIDTH,
            box[2] / Params.IMG_WIDTH, box[3] / Params.IMG_WIDTH]

  def process(self, raw_data, input_data, ground_truth, piped_params=None):
    """
    @pre: Depends on Params.CLASSES
    """
    gt_boxes = []  # gt boxes in normalized image coordinates with cx, cy, width, height in normalized image coordinates
    gt_classes = []  # gt classes, index of Params.CLASSES

    for obj in raw_data["objects"]:
      # obj["bbox"] is stored with [tx, ty, width, height] with [tx, ty] being the top left corner
      top_x = obj["bbox"][0]
      top_y = obj["bbox"][1]
      width = obj["bbox"][2]
      height = obj["bbox"][3]
      bbox = [top_x + width * 0.5, top_y + height * 0.5, width, height]
      bbox, ratio = self._apply_roi(bbox, piped_params["roi"])
      pixels = bbox[2] * bbox[3]
      bbox = self._normalize_box(bbox)
      if obj["obj_class"] in Params.CLASSES and 0.1 < ratio < 10 and pixels > 87:
        gt_classes.append(Params.CLASSES.index(obj["obj_class"]))
        gt_boxes.append(bbox)
      # else:
      #   raise ValueError("Class: " + str(obj["obj_class"]) + " does not exist")

    # show gt boxes in image for debugging
    # self._debug_show_img(input_data, gt_boxes)

    feature_maps, _ = self._prior_boxes.match(gt_boxes, gt_classes, iou_threshold=0.35, iou_min=0.1)

    # data needs to be a 1D array which flattens
    # [feature_map_idx][row][column][ratio_idx][class..., offsets...]
    ground_truth = np.concatenate([fm.flatten() for fm in feature_maps])

    return raw_data, input_data, ground_truth, piped_params
