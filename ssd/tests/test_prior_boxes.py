from ssd.src.params import Params, FeatureMapConfig
from ssd.src.prior_boxes import PriorBoxes
import cv2
import numpy as np
from random import randrange


class TestPriorBoxes:
  def setup_method(self):
    """
    Set up parameters for test methods
    :return: None
    """
    """
    Params.IMG_WIDTH = 500
    Params.IMG_HEIGHT = 400
    Params.FEATURE_MAPS = (
        FeatureMapConfig(size=(9, 7), ratios=[(1.0, 1.0)]),
        FeatureMapConfig(size=(5, 3), ratios=[(0.5, 1.5), (0.5, 1.5)]),
        FeatureMapConfig(size=(2, 2), ratios=[(1.0, 1.0), (1.5, 0.5), (0.5, 1.5)]),
    )
    # Make scales very small, that way it is easier to visualize
    Params.SCALE_MAX = 0.25
    Params.SCALE_MIN = 0.15
    """
    self.prior_boxes = PriorBoxes(clip_boxes=False)

  @staticmethod
  def draw_box(img, box, offset, color):
    x1 = (box[0] - box[2] * 0.5) * Params.IMG_WIDTH + offset
    x2 = (box[0] + box[2] * 0.5) * Params.IMG_WIDTH + offset
    y1 = (box[1] - box[3] * 0.5) * Params.IMG_WIDTH + offset
    y2 = (box[1] + box[3] * 0.5) * Params.IMG_WIDTH + offset
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

  @staticmethod
  def show_boxes(boxes):
    # show prior boxes in
    for i, fm in enumerate(boxes):
      # Note in cv2 it always [height, width]
      offset = int(Params.IMG_WIDTH * 0.2)
      fm_img = np.zeros((Params.IMG_HEIGHT + 2 * offset, Params.IMG_WIDTH + 2 * offset, 3), np.uint8)
      # fill image white
      fm_img[offset:Params.IMG_HEIGHT + offset, offset:Params.IMG_WIDTH + offset] = (255, 255, 255)

      only_show_one_cell = True
      for column_idx, column in enumerate(fm):
        if only_show_one_cell and column_idx > 0:
          break
        for row_idx, fm_position in enumerate(column):
          if only_show_one_cell and row_idx > 0:
            break
          for ratio_idx, box in enumerate(fm_position):
            # rectangle wants [x1, y1] (top-left corner), [x2, y2] (bottom-right corner)
            TestPriorBoxes.draw_box(fm_img, box, offset, (randrange(0, 255, 1), randrange(0, 255, 1),
                                                          randrange(0, 255, 1)))

      cv2.imshow("fm_" + str(i), fm_img)
      cv2.waitKey(0)

  def test_create_prior_boxes(self):
    # assert len(self.prior_boxes.boxes) == 3
    TestPriorBoxes.show_boxes(self.prior_boxes.get_boxes())

  def test_iou(self):
    box_a = np.array([0.5, 0.4, 0.2, 0.3])
    box_b = np.array([0.5, 0.4, 0.2, 0.3])
    iou = self.prior_boxes._calc_iou(box_a, box_b)
    assert iou == 1.0

    box_a = np.array([0.0, 0.0, 0.0, 0.0])
    box_b = np.array([0.2, 0.3, 0.3, 0.4])
    iou = self.prior_boxes._calc_iou(box_a, box_b)
    assert iou == 0.0

    box_a = np.array([0.1, 0.2, 0.2, 0.3])
    box_b = np.array([0.8, 0.7, 0.1, 0.1])
    iou = self.prior_boxes._calc_iou(box_a, box_b)
    assert iou == 0.0

    box_a = np.array([0.3, 0.4, 0.3, 0.3])
    box_b = np.array([0.2, 0.3, 0.3, 0.4])
    iou = self.prior_boxes._calc_iou(box_a, box_b)
    assert iou == 0.3125
