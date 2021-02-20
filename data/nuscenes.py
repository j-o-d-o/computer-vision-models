import argparse
import cv2
import numpy as np
import math
import os
from tqdm import tqdm
from pymongo import MongoClient
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from common.utils import calc_cuboid_from_3d, bbox_from_cuboid, wrap_angle
from data.nuimages_od import CLASS_MAP, Quaternion
from data.od_spec import Object, Entry, OD_CLASS_MAPPING


def main(args):
    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.path, verbose=True)
    nusc.list_scenes()

    for scene in tqdm(nusc.scene):
        next_sample_token = scene["first_sample_token"]

        while True:
            sample = nusc.get('sample', next_sample_token)
            next_sample_token = sample["next"]
            
            sample_data_token = sample["data"]["CAM_FRONT"]
            sample_data = nusc.get_sample_data(sample_data_token)
            cam_front_data = nusc.get('sample_data', sample_data_token)

            # Create image data
            img_path = sample_data[0]
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_bytes = cv2.imencode('.jpeg', img)[1].tobytes()
                content_type = "image/jpeg"
            else:
                print("WARNING: file not found: " + img_path + ", continue with next image")
                continue

            # Get sensor extrinsics, Not sure why roll and yaw seem to be PI/2 off compared to nuImage calibarted sensor
            sensor = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            q = Quaternion(sensor["rotation"][0], sensor["rotation"][1], sensor["rotation"][2], sensor["rotation"][3])
            roll  = math.atan2(2.0 * (q.z * q.y + q.w * q.x) , 1.0 - 2.0 * (q.x * q.x + q.y * q.y)) + math.pi * 0.5
            pitch = math.asin(2.0 * (q.y * q.w - q.z * q.x)) 
            yaw   = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x)) + math.pi * 0.5
            # print(sensor["translation"])
            # print(f"Pitch: {pitch*57.2} Yaw: {yaw*57.2} Roll: {roll*57.2}")

            # Sensor calibration is static, pose would be dynamic. TODO: Somehow also add some sort of cam to cam motion to be learned
            # ego_pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
            # q = Quaternion(ego_pose["rotation"][0], ego_pose["rotation"][1], ego_pose["rotation"][2], ego_pose["rotation"][3])
            # roll  = math.atan2(2.0 * (q.z * q.y + q.w * q.x) , 1.0 - 2.0 * (q.x * q.x + q.y * q.y)) + math.pi * 0.5
            # pitch = math.asin(2.0 * (q.y * q.w - q.z * q.x)) 
            # yaw   = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x)) + math.pi * 0.5
            # print(ego_pose["translation"])
            # print(f"Pitch: {pitch*57.2} Yaw: {yaw*57.2} Roll: {roll*57.2}")

            entry = Entry(
                img=img_bytes,
                content_type=content_type,
                org_source="nuscenes",
                org_id=img_path,
                objects=[],
                ignore=[],
                has_3D_info=True,
                has_track_info=True,
                sensor_valid=True,
                yaw=wrap_angle(yaw),
                roll=wrap_angle(roll),
                pitch=wrap_angle(pitch),
                translation=sensor["translation"],
                scene_token=sample["scene_token"],
                timestamp=sample["timestamp"]
            )

            labels = sample_data[1]
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
                    box3d = calc_cuboid_from_3d(pos_3d, cam_mat, rot_mat, box.wlh[0], box.wlh[2], box.wlh[1])
                    box2d = bbox_from_cuboid(box3d)

                    annotation = nusc.get("sample_annotation", box.token)
                    instance_token = annotation["instance_token"]

                    entry.objects.append(Object(
                        obj_class=CLASS_MAP[box.name],
                        box2d=box2d,
                        box3d=box3d,
                        box3d_valid=True,
                        instance_token=instance_token,
                        truncated=None,
                        occluded=None,
                        width=box.wlh[0],
                        length=box.wlh[1],
                        height=box.wlh[2],
                        orientation=rot_angle,
                        # converted to autosar coordinate system
                        x=pos_3d[2],
                        y=-pos_3d[0],
                        z=-pos_3d[1]
                    ))

                    # For debugging show the data in the image
                    # box2d = list(map(int, box2d))
                    # cv2.circle(img, (int(pos_2d[0]), int(pos_2d[1])), 3, (255, 0, 0))
                    # cv2.rectangle(img, (box2d[0], box2d[1]), (box2d[0] + box2d[2], box2d[1] + box2d[3]), (255, 255, 0), 1)

            # img = cv2.resize(img, (800, 460))
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            collection.insert_one(entry.get_dict())

            if next_sample_token == "":
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload 2D and 3D data from nuscenes dataset")
    parser.add_argument("--path", type=str, help="Path to nuscenes data, should contain samples/CAMERA_FRONT/*.jpg and v1.0-trainval/*.json folder e.g. /path/to/nuscenes")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuscenes", help="MongoDB collection")

    main(parser.parse_args())
