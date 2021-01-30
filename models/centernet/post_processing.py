import numpy as np
from common.utils import Roi, convert_back_to_roi
from models.centernet.params import Params


def process_2d_output(output_mask, roi: Roi, params: Params, min_conf_value = 0.25):
    """
    Sliding window to find maximas which are related to objects
    :param output_mask: output of the centernet
    :param roi: region of interesst of input compared to org image
    :param params: information of active output and indices of the values in the output_mask as well as R scaling
    :param min_conf_value: every peak above this threshold will be considered an object
    """
    objects = []
    class_mask = output_mask[:, :, : params.NB_CLASSES]

    # window size in (y, x)
    window_size = np.array((9, 9), dtype=np.int64)
    window_center = np.int64(np.floor(window_size * 0.5))

    output_shape = output_mask.shape
    # loop over every pixel per class
    for y, x, cls_idx in np.ndindex((output_shape[0] - window_center[0], output_shape[1] - window_center[1], params.NB_CLASSES)):
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
                # mandatory fields, all others are optional depending on active regression fields
                obj = {"cls_idx": cls_idx, "center": convert_back_to_roi(roi, [x * params.R, y * params.R])}

                if params.REGRESSION_FIELDS["r_offset"]:
                    r_offset = curr_pixel[params.start_idx("r_offset"):params.end_idx("r_offset")]
                    obj["center"] = convert_back_to_roi(roi, [(x + r_offset[0]) * params.R, (y + r_offset[1]) * params.R])
                
                if params.REGRESSION_FIELDS["fullbox"]:
                    fullbox = curr_pixel[params.start_idx("fullbox"):params.end_idx("fullbox")]
                    width = fullbox[0] * (1.0 / roi.scale)
                    height = fullbox[1] * (1.0 / roi.scale)
                    # fill as [top_left_x, top_left_y, width, height]
                    obj["fullbox"] = [obj["center"][0] - (width / 2.0), obj["center"][1] - (height / 2.0), width, height]

                if params.REGRESSION_FIELDS["l_shape"]:
                    l_shape = curr_pixel[params.start_idx("l_shape"):params.end_idx("l_shape")]
                    obj["bottom_left"] = obj["center"] + l_shape[0: 2] * (1.0 / roi.scale)
                    obj["bottom_right"] = obj["center"] + l_shape[2: 4] * (1.0 / roi.scale)
                    obj["bottom_center"] = obj["center"] + l_shape[4: 6] * (1.0 / roi.scale)
                    obj["center_height"] = l_shape[6] * (1.0 / roi.scale)

                if params.REGRESSION_FIELDS["l_shape"]:
                    l_shape = curr_pixel[params.start_idx("l_shape"):params.end_idx("l_shape")]
                    obj["bottom_left"] = obj["center"] + l_shape[0: 2] * (1.0 / roi.scale)
                    obj["bottom_right"] = obj["center"] + l_shape[2: 4] * (1.0 / roi.scale)
                    obj["bottom_center"] = obj["center"] + l_shape[4: 6] * (1.0 / roi.scale)
                    obj["center_height"] = l_shape[6] * (1.0 / roi.scale)

                if params.REGRESSION_FIELDS["3d_info"]:
                    l_shape = curr_pixel[params.start_idx("3d_info"):params.end_idx("3d_info")]
                    # TODO: fill 3d info to object

                objects.append(obj)
    return objects
