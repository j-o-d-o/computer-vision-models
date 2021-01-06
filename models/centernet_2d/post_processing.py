import numpy as np
from common.utils import Roi, convert_to_roi


def process_2d_output(output_mask, roi: Roi, r: float, nb_classes: int):
    """
    Sliding window to find maximas which are related to objects
    :param output_mask: output of the centernet, assumed encoding on last axis:
    :                   [0:nb_classes]: classes, [nb_classes]: offset_x, [nb_classes+1]: offset_y, [nb_classes+2]: width, [nb_classes+3]: height, 
    :param roi: region of interesst of input compared to org image
    :param r: scale of output_mask compared to input
    :param nb_classes: number of classes that are used in the output_mask
    """
    objects = []
    class_mask = output_mask[:, :, :nb_classes]

    # window size in (y, x)
    window_size = np.array((5, 5), dtype=np.int64)
    window_center = np.int64(np.floor(window_size * 0.5))

    # min confidence the maximum needs to have
    min_conf_value = 0.25

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
                r_scaled_center = ((x * r) + offset_x, (y * r) + offset_y)
                center = convert_to_roi(roi, r_scaled_center)
                width = curr_pixel[nb_classes + 2] * r * (1 / roi.scale)
                height = curr_pixel[nb_classes + 3] * r * (1 / roi.scale)
                # location, width and height in relation to the org image size
                objects.append({
                    "obj_idx": cls_idx,
                    "top_left": (int(center[0] - (width * 0.5)), int(center[1] - (height * 0.5))),
                    "bottom_right": (int(center[0] + (width * 0.5)), int(center[1] + (height * 0.5)))
                })
    return objects
