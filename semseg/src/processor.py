import numpy as np
import cv2
from dataclasses import dataclass
from semseg.src.params import Params
from data.semseg_spec import SEMSEG_CLASS_MAPPING
from mlpipe.processors.i_processor import IPreProcessor
from tensorflow.keras.utils import to_categorical


@dataclass
class Roi:
  # offsets are pre scaling!
  offset_top: int = 0
  offset_bottom: int = 0
  offset_left: int = 0
  offset_right: int = 0
  scale: float = 1.0


def to_hex(array):
  array = np.asarray(array, dtype='uint32')
  return (array[:, :, 0] << 16) + (array[:, :, 1] << 8) + array[:, :, 2]


def key_to_index(value):
  idx = 0
  for cls, colour in SEMSEG_CLASS_MAPPING.items():
    if value == colour:
      return idx
    idx += 1
  assert(False and "colour does not exist: " + str(value))
  return -1


class ProcessImages(IPreProcessor):
  @staticmethod
  def resize_img(img: np.ndarray, goal_width: float, goal_height: float, offset_top: float = None, interpolation: int = cv2.INTER_LINEAR) -> (np.ndarray, Roi):
    """
    Resize image in a way that it fits the params, the default cropping will take delta height from top and delta width
    from left and right border equally
    :param img: numpy img array (as used by cv2), note that it will also be changed in place
    :param goal_width: width the image should have after resizing
    :param goal_height: height the image should have after resizing
    :param offset_top: if height has to be cropped, what offset should be used from top. In case of None all delta will be cropped from top
    :param interpolation: Interpolation which should be used, default is cv2.INTER_LINEAR
    :return: scaled and cropped image, roi data
    """
    # offsetting is done prior to scaling!
    roi_data = Roi()

    h, w = img.shape[:2]
    curr_ratio = w / h
    target_ratio = goal_width / goal_height
    if curr_ratio > target_ratio:
      # cut width equally from left and right edge
      delta_width = int((w - (target_ratio * h)) / 2)
      img = img[:, delta_width:(w-delta_width)]
      roi_data.offset_left += delta_width
      roi_data.offset_right += delta_width
    else:
      delta_height = int(h - (w / target_ratio))
      if offset_top is None:
        offset_top = delta_height
        offset_bottom = 0
      else:
        offset_bottom = delta_height - offset_top
        assert(offset_bottom >= 0)
      img = img[offset_top:(h-offset_bottom)]
      roi_data.offset_top += offset_top
      roi_data.offset_bottom += offset_bottom
    unscaled_h, unscaled_w = img.shape[:2]
    roi_data.scale = goal_width / unscaled_w
    img = cv2.resize(img, (goal_width, goal_height), interpolation=interpolation)
    return img, roi_data

  def process(self, raw_data, input_data, ground_truth, piped_params=None):
    # Add input_data
    img_encoded = np.frombuffer(raw_data["img"], np.uint8)
    input_data = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    input_data, roi_img = self.resize_img(input_data, Params.IMG_WIDTH, Params.IMG_HEIGHT, offset_top=Params.OFFSET_TOP)
    input_data = input_data.astype(np.float32)
    piped_params["roi_img"] = roi_img

    # Add ground_truth mask
    mask_encoded = np.frombuffer(raw_data["mask"], np.uint8)
    mask_img = cv2.imdecode(mask_encoded, cv2.IMREAD_COLOR)
    mask_img, roi_img = self.resize_img(mask_img, Params.MASK_WIDTH, Params.MASK_HEIGHT, offset_top=Params.OFFSET_TOP, interpolation=cv2.INTER_NEAREST)
    piped_params["roi_img"] = roi_img
    # one hot encode based on class mapping from Params
    mask_img = to_hex(mask_img) # convert 3 channel representation to single hex channel
    vfunc = np.vectorize(key_to_index)
    mask_img = vfunc(mask_img)

    nb_classes = len(SEMSEG_CLASS_MAPPING)
    ground_truth = to_categorical(mask_img, nb_classes)

    return raw_data, input_data, ground_truth, piped_params
