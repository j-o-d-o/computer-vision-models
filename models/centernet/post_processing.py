import numpy as np
from common.utils import Roi, convert_back_to_roi


def process_2d_output(output_mask, roi: Roi, r: float, nb_classes: int, min_conf_value = 0.25):
    """
    Sliding window to find maximas which are related to objects
    :param output_mask: output of the centernet, assumed encoding on last axis:
    :                   [0:nb_classes]: classes, (following indices have all offset nb_classes) [0:1]: loc_offset, [2]: width px, [3]: height px,
    :                   [4:5]: bottom_left_off, [6:7]: bottom_right_off, [8:9]: bottom_center_off, [10]: center_height, [11]: radial_distance,
    :                   [12]: orientation, [13]: width, [14]: height, [15]: length
    :param roi: region of interesst of input compared to org image
    :param r: scale of output_mask compared to input
    :param nb_classes: number of classes that are used in the output_mask
    :param min_conf_value: every peak above this threshold will be considered an object
    """
    objects = []
    class_mask = output_mask[:, :, :nb_classes]

    # window size in (y, x)
    window_size = np.array((5, 5), dtype=np.int64)
    window_center = np.int64(np.floor(window_size * 0.5))

    output_shape = output_mask.shape
    # loop over every pixel per class
    for y, x, cls_idx in np.ndindex((output_shape[0] - window_center[0], output_shape[1] - window_center[1], nb_classes)):
        if y >= window_center[0] and x >= window_center[1]:
            # get values for the current window
            start_y = y - window_center[0]
            end_y = y + window_center[0] + 1
            start_x = x - window_center[1]
            end_x = x + window_center[1] + 1
            window_values = output_mask[start_y:end_y, start_x:end_x, cls_idx]
            # find max_idx of the window as tuple
            max_idx = np.unravel_index(np.argmax(window_values), window_size)
            curr_pixel = output_mask[y][x]
            # if maximum relates to window center and the confidence exeeds threshold, save as object
            if max_idx == tuple(window_center) and curr_pixel[cls_idx] > min_conf_value:
                offset_x = curr_pixel[nb_classes]
                offset_y = curr_pixel[nb_classes + 1]
                center = convert_back_to_roi(roi, [(x + offset_x) * r, (y + offset_y) * r])
                width_px = curr_pixel[nb_classes + 2] * (1 / roi.scale)
                height_px = curr_pixel[nb_classes + 3] * (1 / roi.scale)
                objects.append({
                    "cls_idx": cls_idx,
                    "center": center,
                    "fullbox": [center[0] - (width_px / 2.0), center[1] - (height_px / 2.0), width_px, height_px],
                    "bottom_left": center + curr_pixel[nb_classes + 4: nb_classes + 6] * (1 / roi.scale),
                    "bottom_right": center + curr_pixel[nb_classes + 6: nb_classes + 8] * (1 / roi.scale),
                    "bottom_center": center + curr_pixel[nb_classes + 8: nb_classes + 10] * (1 / roi.scale),
                    "center_height": curr_pixel[nb_classes + 10] * (1 / roi.scale)
                })

    return objects
