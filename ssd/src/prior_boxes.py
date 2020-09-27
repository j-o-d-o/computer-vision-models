import tensorflow as tf
import numpy as np
import json
from typing import List
from ssd.src.params import Params


class PriorBoxes:
  """
  Class to create PriorBoxes with [x, y, width, height) with [x, y] being the center of the box. Normalized [0, 1]
  image coordinates are used
  """

  def __init__(self, clip_boxes: bool = True):
    """
    :param clip_boxes: if true, prior boxes can not go beyond the image borders
    """
    self._clip_boxes = clip_boxes
    self._boxes: List[np.ndarray] = []
    self._num_boxes: int = 0
    self._init_boxes()

  def get_num_boxes(self):
    return self._num_boxes

  def get_boxes(self):
    return self._boxes

  def get_flattened_boxes(self):
    return np.concatenate([fm.flatten() for fm in self._boxes])

  def save_to_file(self, file_path):
    flattened_pb = self.get_flattened_boxes().tolist()
    json.dump(flattened_pb, open(file_path, 'w'))

  def _init_boxes(self) -> None:
    """
    Create the prior boxes
    @pre: Depends on Params.FEATURE_MAPS
    """
    for i, feature_map_info in enumerate(Params.FEATURE_MAPS):
      # coordinates are normalized to IMG_WIDTH, thus there needs to be an extra scaling for height
      height_scale = Params.IMG_HEIGHT / Params.IMG_WIDTH
      ratios = feature_map_info.ratios
      nb_boxes = len(ratios)
      fm_height = feature_map_info.size[0]
      fm_width = feature_map_info.size[1]
      # I do not like the scale of the paper
      # scale = PriorBoxes._calc_scale(i + 1)
      # I'd rather have the scale so that a 1.0 in the feature map is equal to one "cell" of the feature map
      scale = 1 / fm_width
      # e.g. 8 x 8 (feature map size) x 6 (different ratios) x 4 (box coordinates)
      prior_boxes = np.zeros((fm_height, fm_width, nb_boxes, 4))

      for row_idx, column in enumerate(prior_boxes):
        for column_idx, fm_ratios in enumerate(column):
          center_x = (float(column_idx) + 0.5) * (1.0 / float(fm_width))
          center_y = (float(row_idx) + 0.5) * (1.0 / float(fm_height)) * height_scale
          for x, ratio in enumerate(ratios):
            width = ratio[0] * scale
            height = ratio[1] * scale

            # height_scale = the maximum height in normalized coordinates
            if self._clip_boxes:
              # test if box would be beyond image borders in x-direction
              if center_x - 0.5 * width < 0.0:
                cut = abs(center_x - 0.5 * width)
                width -= cut
                center_x += cut * 0.5
              if center_x + 0.5 * width > 1.0:
                cut = (center_x + 0.5 * width) - 1.0
                width -= cut
                center_x -= cut * 0.5
              if center_y - 0.5 * height < 0.0:
                cut = abs(center_y - 0.5 * height)
                height -= cut
                center_y += cut * 0.5
              if center_y + 0.5 * height > height_scale:
                cut = (center_y + 0.5 * height) - height_scale
                height -= cut
                center_y -= cut * 0.5

            fm_ratios[x] = np.array([center_x, center_y, width, height])
            self._num_boxes += 1

      self._boxes.append(prior_boxes)

  @staticmethod
  def _calc_scale(k: int) -> float:
    """
    From paper: https://arxiv.org/pdf/1512.02325.pdf Choosing scales and aspect ratios for default boxes
    @pre: Depends on Params.SCALE_MIN, Params.SCALE_MAX, len(Params.FEATURE_MAPS)
    :param k: number of feature map (counting starts at 1 with the largest feature map in terms of pixels)
    :return: float value of scale that should be used with the feature map
    """
    m = float(len(Params.FEATURE_MAPS))
    k = float(k)
    s_min = Params.SCALE_MIN
    s_max = Params.SCALE_MAX
    scale = s_min + ((s_max - s_min) / (m - 1.0)) * (k - 1.0)
    return scale

  @staticmethod
  def _calc_iou(box_a: np.ndarray, box_b: np.ndarray, type_a: str = "center", type_b: str = "center") -> float:
    """
    Calculate IOU of two boxes. Both boxes should be in in normalized [0, 1] image coordinates
    :param box_a: rectangular box
    :param box_b: rectangular box
    :param box_a: center -or- corner, center has cx, cy, width, height, corner has x1, y1, x2, y2
    :return: float value of the iou
    """
    if type_a == "center":
      # convert to corner format
      box_a = np.array([
        box_a[0] - 0.5 * box_a[2],
        box_a[1] - 0.5 * box_a[3],
        box_a[0] + 0.5 * box_a[2],
        box_a[1] + 0.5 * box_a[3],
      ])
    if type_b == "center":
      # convert to corner format
      box_b = np.array([
        box_b[0] - 0.5 * box_b[2],
        box_b[1] - 0.5 * box_b[3],
        box_b[0] + 0.5 * box_b[2],
        box_b[1] + 0.5 * box_b[3],
      ])
    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return float(iou)

  def match(self, gt_boxes: list, gt_classes: list, iou_threshold: float = 0.01, iou_min: float = 0.01) -> (
      List[np.ndarray], List[dict]):
    """
    Match boxes
    :param gt_boxes: numpy array of boxes e.g. ground truth boxes (format cx, cy, width, height) in normalized
                     (to width) image coordinates [0, 1]
    :param gt_classes: numpy array of class idx for the ground truth class (must be same length as gt_boxes)
    :param iou_threshold: threshold of iou which a box is considered a match
    :return: lists of numpy arrays with [fm_idx][height][width][ratio_idx] = [class.., offsets...],
             iou_map with [gt_box_idx]{iou: [], box} with i indexing all matched prior box ious
    """
    iou_map = []
    for _ in gt_boxes:
      iou_map.append({"iou": [], "box": None})

    fms_gt: List[np.ndarray] = [] # ground truth feature map storing [*class, *offsets] for each prior box
    assert len(gt_boxes) == len(gt_classes), "gt_boxes and gt_classes need to be index parallel!"
    assert iou_threshold >= 0.0, "iou_threshold should not be bellow 0!"
    iou_threshold = max(iou_threshold, 0.0)

    # (score, feature_map_idx, fm_row, fm_column, fm_ratio, class, x, y, width, height)
    best_gt_matches = np.zeros((len(gt_boxes), 10))

    for fm_idx, fm_prior_boxes in enumerate(self._boxes):
      fm_height = fm_prior_boxes.shape[0]
      fm_width = fm_prior_boxes.shape[1]
      nb_ratios = fm_prior_boxes.shape[2]
      # len(Params.CLASSES) + 4 => [*class, *offsets] with offsets in normalized (to width) image coordinates
      fm_gt = np.zeros((fm_height, fm_width, nb_ratios, len(Params.CLASSES) + 4), np.float32)

      # loop through each prior box within the feature map and find the best matching gt_box
      for row_idx, row in enumerate(fm_prior_boxes):
        for column_idx, fm_ratios in enumerate(row):
          for ratio_idx, prior_box in enumerate(fm_ratios):
            # find best matching box
            matched_offset: np.ndarray = np.zeros(4, np.float) # array of 4 floats in normalized image coordinates
            matched_class: int = Params.CLASSES.index("BACKGROUND") # class idx as seen in Params.CLASSES
            max_iou: float = 0.0 # current maximum iou
            matched_gt_box_idx: int = -1
            for gt_box_idx, gt_box in enumerate(gt_boxes):
              iou: float = PriorBoxes._calc_iou(prior_box, gt_box)
              gt_box_pixels = gt_box[2] * Params.IMG_WIDTH * gt_box[3] * Params.IMG_WIDTH # coordinates are normalized to IMG_WIDTH
              if iou > max_iou:
                max_iou = iou
                matched_offset = gt_box - prior_box
                matched_gt_box_idx = gt_box_idx
                matched_class = int(gt_classes[gt_box_idx])

            if matched_gt_box_idx >= 0 and max_iou > best_gt_matches[matched_gt_box_idx][0]:
              # new best match for this gt box -> save it
              best_gt_matches[matched_gt_box_idx] = [
                max_iou,
                fm_idx, row_idx, column_idx, ratio_idx,
                matched_class,
                *matched_offset,
              ]

            class_map = np.zeros((len(Params.CLASSES)))
            if max_iou > iou_threshold:
              class_map[matched_class] = 1.0
              iou_map[matched_gt_box_idx]["iou"].append(max_iou)
              iou_map[matched_gt_box_idx]["box"] = gt_boxes[matched_gt_box_idx]
            else:
              class_map[Params.CLASSES.index("BACKGROUND")] = 1.0
              matched_offset = np.zeros(4, np.float)

            fm_gt[row_idx][column_idx][ratio_idx] = [*class_map, *matched_offset]

      fms_gt.append(fm_gt)

    # every object should have at least one match (even when below the threshold set)
    force = 0
    non_force = 0
    for gt_box_idx, match in enumerate(best_gt_matches):
      if iou_min <= match[0] <= iou_threshold:
        force += 1
        # print("FORCING WITH: " + str(match[0]))
        # force this prior box to the object
        fm_idx = int(match[1])
        row_idx = int(match[2])
        column_idx = int(match[3])
        ratio_idx = int(match[4])
        class_idx = int(match[5])
        class_map = np.zeros((len(Params.CLASSES)))
        class_map[class_idx] = 1.0
        offset = match[6:]
        fms_gt[fm_idx][row_idx][column_idx][ratio_idx] = [*class_map, *offset]
        iou_map[gt_box_idx]["iou"].append(match[0])
        iou_map[gt_box_idx]["box"] = gt_boxes[gt_box_idx]
      else:
        non_force += 1

    # print("GOOD | FORCE: " + str(non_force) + " | " + str(force))
    return fms_gt, iou_map


class PriorBoxDecoder:
  def __init__(self, boxes: np.ndarray = None):
    """
    :param boxes: Flattened list of prior boxes of size nb_boxes * 4. Box in [cx, cy, w, h] in normalized img coords
    """
    if boxes is not None:
      self._boxes = boxes.reshape((-1, 4))

  def init_boxes_from_file(self, file_path: str):
    with open(file_path, "r") as json_file:
      data = json.load(json_file)
      self._boxes = np.array(data).reshape((-1, 4))

  @staticmethod
  def normalize(x: np.ndarray):
    x = tf.math.softmax(x)
    return x.numpy()

  def decode_boxes(self, output: np.array, num_classes: int, threshold: float = 0.1):
    """
    Decode the detected boxes from the offset and class output of the SSD net
    :param output: 1D-numpy-array of [classes..., offsets...] for each prior box (starting from largest to smallest
                   prior box)
    :param num_classes: number of classes per prior box
    :param threshold: Background threshold for which it is marked as "detection" (otherwise it is just background)
    :return: list of boxes [box_idx][[cx, cy, width, height],[CLASS_PROBABILITIES...]]
    """
    output = output.reshape((-1, (num_classes + 4)))
    prior_box_offsets = output[:, num_classes:]
    prior_box_classes = output[:, :-4]

    # offsets, classes and prior boxes need to have the same length
    assert prior_box_offsets.shape[0] == prior_box_classes.shape[0] == self._boxes.shape[0]

    detected_boxes = []
    for off, cls, pb in zip(prior_box_offsets, prior_box_classes, self._boxes):
      # norm_classes = cls / sum(cls)
      norm_classes = self.normalize(cls).round(2)
      is_obj_score = np.max(norm_classes[1:])
      if is_obj_score >= threshold:
        detected_boxes.append([pb + off, norm_classes])

    return detected_boxes
