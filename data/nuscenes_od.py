import argparse
import cv2
import numpy as np
import math
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from common.utils import calc_cuboid_from_3d, bbox_from_cuboid, wrap_angle

# Mapping nuscenes classes to od_spec classes
# TODO: bicycle and motorcycle should be merged with closes ped label
CLASS_MAP = {
    "human.pedestrian.adult": "ped",
    "human.pedestrian.child": "ped",
    "human.pedestrian.construction_worker": "ped",
    "human.pedestrian.personal_mobility": "ped",
    "human.pedestrian.police_officer": "ped",
    "human.pedestrian.stroller": "ped",
    "human.pedestrian.wheelchair": "ped",
    "vehicle.bicycle": "cyclist",
    "vehicle.motorcycle": "motorbike",
    "vehicle.car": "car",
    "vehicle.emergency.police": "car",
    "vehicle.emergency.ambulance": "van",
    "vehicle.construction": "truck",
    "vehicle.bus.bendy": "truck",
    "vehicle.bus.rigid": "truck",
    "vehicle.trailer": "truck",
    "vehicle.truck": "truck"
}
# Dont care classes (will be added as ignore areas to the dataset)
DONT_CARE_CLASSES = ["animal"]
# Ignore these kitti classes, note DontCare will be added to the ignore areas of the od spec
IGNORE_CLASSES = ["movable_object.barrier", "movable_object.debris", "movable_object.pushable_pullable",
 "movable_object.trafficcone", "static_object.bicycle_rack"]


def main(args):
    args.path = "/home/jo/training_data/nuscenes/nuimages-v1.0-mini"

    nusc = NuScenes(version="v1.0-mini", dataroot=args.path, verbose=True)
    nusc.list_scenes()

    for scene in nusc.scene:
        next_sample_token = scene["first_sample_token"]

        while True:
            sample = nusc.get('sample', next_sample_token)
            next_sample_token = sample["next"]
            sample_data_token = sample["data"]["CAM_FRONT"]
            sample_data = nusc.get_sample_data(sample_data_token)

            data_path = sample_data[0]
            img = cv2.imread(data_path)

            labels = sample_data[1]
            debug_img = img.copy()

            for box in labels:
                if box.name in CLASS_MAP.keys():
                    box.translate(np.array([0, box.wlh[2] / 2, 0])) # translate center center to bottom center
                    pos_3d = np.array([*box.center, 1.0])
                    cam_mat = np.hstack((sample_data[2], np.zeros((3, 1))))
                    # create rotation matrix, somehow using the rotation matrix directly from nuscenes does not work
                    # instead, we calc the rotation as it is in kitti and use the same code
                    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                    yaw = -np.arctan2(v[2], v[0])
                    rot_angle = wrap_angle(float(yaw) + math.pi * 0.5) # because parallel to optical view of camera = 90 deg
                    rot_mat = np.array([
                        [math.cos(rot_angle), 0.0, math.sin(rot_angle), 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-math.sin(rot_angle), 0.0, math.cos(rot_angle), 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    pos_2d = np.matmul(cam_mat, pos_3d)
                    pos_2d /= pos_2d[2]
                    cv2.circle(debug_img, (int(pos_2d[0]), int(pos_2d[1])), 3, (255, 0, 0))
                    box3d = calc_cuboid_from_3d(pos_3d, cam_mat, rot_mat, box.wlh[0], box.wlh[2], box.wlh[1], debug_img)
                    box2d = bbox_from_cuboid(box3d)
                    box2d = list(map(int, box2d))
                    cv2.rectangle(debug_img, (box2d[0], box2d[1]), (box2d[0] + box2d[2], box2d[1] + box2d[3]), (255, 255, 0), 1)

            cv2.imshow("test", debug_img)
            cv2.waitKey(0)

            if next_sample_token == "":
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload 2D and 3D data from nuscenes dataset")
    parser.add_argument("--path", type=str, help="Path to nuscenes data, should contain samples/CAMERA_FRONT/*.jpg and v1.0-trainval/*.json folder e.g. /path/to/nuscenes")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuscenes", help="MongoDB collection")

    main(parser.parse_args())
