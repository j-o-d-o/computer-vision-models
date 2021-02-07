import cv2
from pymongo import MongoClient
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from data.label_spec import Entry


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

                factor = (640/img_data.shape[1])
                img_data = cv2.resize(img_data, None, img_data, fx=factor, fy=factor)
                depth_data = cv2.resize(depth_data, None, depth_data, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
                top_offset = img_data.shape[0] - 256
                img_data = img_data[top_offset:, :]
                depth_data = depth_data[top_offset:, :]
                
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
