import cv2
from pymongo import MongoClient
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from data.label_spec import Entry
import numpy as np
from numba import jit


@jit(nopython=True)
def fill_depth_data(depth_data):
    # depth_data is very sparse, lets change this
    window_size = np.array((3, 3), dtype=np.int64)
    window_center = np.array([1, 1])
    fill_depth_data = np.zeros(depth_data.shape, dtype=np.uint16)
    # loop over every pixel per class
    for y in range(depth_data.shape[0] - 1):
        for x in range(depth_data.shape[1] - 1):
            if y >= window_center[0] and x >= window_center[1] and depth_data[window_center[0]][window_center[1]] == 0:
                # get values for the current window
                start_y = y - window_center[0]
                end_y = y + window_center[0] + 1
                start_x = x - window_center[1]
                end_x = x + window_center[1] + 1
                window_values = depth_data[start_y:end_y, start_x:end_x]
                mean = None
                count = 0
                for iy, ix in np.ndindex(window_values.shape):
                    value = window_values[iy][ix]
                    if value > 0:
                        if mean is None:
                            mean = value
                        else:
                            mean += value
                            count += 1
                if count > 0:
                    mean /= float(count)
                else:
                    mean = 0
                fill_depth_data[y][x] = mean

    return depth_data + fill_depth_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload semseg data from comma10k dataset")
    parser.add_argument("--depth_map", type=str, help="Path to depth maps")
    parser.add_argument("--images", type=str, help="Path to images")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="depth", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="driving_stereo", help="MongoDB collection")
    args = parser.parse_args()

    args.depth_map = "/home/jo/training_data/drivingstereo/depth_map"
    args.images = "/home/jo/training_data/drivingstereo/left_img"

    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    # calib_root_path = "/home/jo/Downloads/half-image-calib"
    # _, _, filenames = next(os.walk(calib_root_path))
    # for filename in filenames:
    #     with open(calib_root_path + "/" + filename) as f:
    #         # Create calibration matrix for P2 image from calibration data
    #         calib_lines = f.readlines()
    #         calib_lines = [line.strip().split(" ") for line in calib_lines if line.strip()]

    #         calib_101 = calib_lines[3]
    #         focal_length_101 = (float(calib_101[1]), float(calib_101[5]))
    #         pp_offset_101 = (float(calib_101[3]), float(calib_101[6]))
            
    #         calib_103 = calib_lines[11]
    #         focal_length_103 = (float(calib_103[1]), float(calib_103[5]))
    #         pp_offset_103 = (float(calib_103[3]), float(calib_103[6]))

    #         print(f"------- {filename} -------------")
    #         print(focal_length_101)
    #         print(focal_length_103)
    #         print(pp_offset_101)
    #         print(pp_offset_103)

    for folder in tqdm(next(os.walk(args.depth_map))[1]):
        curr_scene_depth_map_root = os.path.join(args.depth_map, folder)
        curr_scene_image_root = os.path.join(args.images, folder)
        _, _, filenames = next(os.walk(curr_scene_depth_map_root))
        filenames.sort() # since we have video frames
        timestamp = 0
        next_timestamp = 1
        for filename in tqdm(filenames):
            depth_file = os.path.join(curr_scene_depth_map_root, filename)
            img_file = os.path.join(curr_scene_image_root, filename[:-3] + "jpg")
            if os.path.isfile(img_file):
                img_data = cv2.imread(img_file)
                depth_data = cv2.imread(depth_file, -1) # load as is (uint16 grayscale img)
                depth_data_org = depth_data
                depth_data = fill_depth_data(depth_data)

                factor = (640/img_data.shape[1])
                img_data = cv2.resize(img_data, None, img_data, fx=factor, fy=factor)
                depth_data = cv2.resize(depth_data, None, depth_data, fx=(factor*0.5), fy=(factor*0.5), interpolation=cv2.INTER_NEAREST)
                top_offset = img_data.shape[0] - 256
                img_data = img_data[top_offset:, :]
                depth_data = depth_data[(top_offset//2)+1:, :]

                img_bytes = cv2.imencode(".jpg", img_data)[1].tobytes()
                depth_bytes = cv2.imencode(".png", depth_data)[1].tobytes()

                entry = Entry(
                    img=img_bytes,
                    depth=depth_bytes,
                    content_type="image/jpg",
                    org_source="driving_stereo",
                    org_id=filename[:-3],
                    scene_token=folder,
                    timestamp=timestamp,
                    next_timestamp=next_timestamp
                )
                collection.insert_one(entry.get_dict())

                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
                # ax2.imshow(cv2.cvtColor(depth_data, cv2.COLOR_BGR2RGB))
                # plt.show()

                timestamp += 1
                if next_timestamp is not None:
                    next_timestamp += 1
                    if next_timestamp == len(filenames):
                        next_timestamp = None
            else:
                print(f"ERROR: {img_file}")
