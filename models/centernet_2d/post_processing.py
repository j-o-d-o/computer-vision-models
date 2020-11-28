import numpy as np
from common.utils import Roi, convert_to_roi


def process_2d_output(output_mask, roi: Roi, r: float, nb_classes: int):
    objects = []
    class_mask = output_mask[:, :, :nb_classes]

    output_shape = output_mask.shape
    for y, x, cls_idx in np.ndindex((output_shape[0], output_shape[1], nb_classes)):
        if y >= 2 and y < (output_shape[0] - 1) and x >= 2 and x <= (output_shape[1] - 1):
            start_y = y - 2
            end_y = y + 3
            start_x = x - 2
            end_x = x + 3
            max_idx = np.unravel_index(np.argmax(output_mask[start_y:end_y, start_x:end_x, cls_idx]), (5, 5))
            curr_pixel = output_mask[y][x]
            if max_idx == (2, 2) and curr_pixel[cls_idx] > 0.25:
                # found object
                offset_x = curr_pixel[nb_classes]
                offset_y = curr_pixel[nb_classes + 1]
                r_scaled_center = ((x * r) + offset_x, (y * r) + offset_y)
                center = convert_to_roi(roi, r_scaled_center)
                width = curr_pixel[nb_classes + 2] * r * (1 / roi.scale)
                height = curr_pixel[nb_classes + 3] * r * (1 / roi.scale)
                objects.append({
                    "obj_idx": cls_idx,
                    "top_left": (int(center[0] - (width * 0.5)), int(center[1] - (height * 0.5))),
                    "bottom_right": (int(center[0] + (width * 0.5)), int(center[1] + (height * 0.5)))
                })

    return objects
