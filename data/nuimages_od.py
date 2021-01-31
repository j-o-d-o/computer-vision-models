import argparse
import cv2
import numpy as np
import math
import os
from dataclasses import dataclass
from tqdm import tqdm
from pymongo import MongoClient
from nuimages import NuImages
from nuscenes.utils.data_classes import Box
from data.od_spec import Object, Entry, OD_CLASS_MAPPING
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
# Ignore these kitti classes, note DontCare will be added to the ignore areas of the od spec
IGNORE_CLASSES = ["movable_object.barrier", "movable_object.debris", "movable_object.pushable_pullable",
 "movable_object.trafficcone", "static_object.bicycle_rack", "animal"]

@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

def main(args):
    client = MongoClient(args.conn)
    collection = client[args.db][args.collection]

    args.path = "/home/jo/training_data/nuscenes/nuimages-v1.0-all"
    nuim = NuImages(version="v1.0-test", dataroot=args.path, lazy=True, verbose=False)

    for sample in tqdm(nuim.sample):
        sample_data = nuim.get("sample_data", sample["key_camera_token"])

        # check if already exists, if yes, continue with next
        # check_db_entry = collection.find_one({ "org_source": "nuscenes", "org_id": sample_data["filename"]})
        # if check_db_entry is not None:
        #     print("WARNING: Entry " + str(sample_data["filename"]) + " already exists, continue with next image")
        #     continue

        # Create image data
        img_path = args.path  + "/" + sample_data["filename"]
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img_bytes = cv2.imencode('.jpeg', img)[1].tobytes()
            content_type = "image/jpeg"
        else:
            print("WARNING: file not found: " + img_path + ", continue with next image")
            continue

        # Get sensor extrinsics
        sensor = nuim.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
        q = Quaternion(sensor["rotation"][0], sensor["rotation"][1], sensor["rotation"][2], sensor["rotation"][3])
        roll  = math.atan2(2.0 * (q.z * q.y + q.w * q.x) , 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(2.0 * (q.y * q.w - q.z * q.x))
        yaw   = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x))

        entry = Entry(
            img=img_bytes,
            content_type=content_type,
            org_source="nuscenes",
            org_id=sample_data["filename"],
            objects=[],
            ignore=[],
            has_3D_info=False,
            has_track_info=False,
            sensor_valid=True,
            yaw=wrap_angle(yaw),
            roll=wrap_angle(roll),
            pitch=wrap_angle(pitch),
            translation=sensor["translation"]
        )

        # Create objects
        obj_tokens, surface_tokens = nuim.list_anns(sample['token'], verbose=False)
        for obj_token in obj_tokens:
            obj = nuim.get("object_ann", obj_token)
            obj_cat = nuim.get("category", obj["category_token"])
            nu_class_name = obj_cat["name"]
            if nu_class_name not in IGNORE_CLASSES and nu_class_name in CLASS_MAP:
                obj_class = CLASS_MAP[nu_class_name]
                if obj_class not in list(OD_CLASS_MAPPING.keys()):
                    print("WARNING: Unkown class " + obj_class)
                    continue
                # TODO: Let's use attributes somehow e.g. car.moving
                # for attrib_token in obj["attribute_tokens"]:
                #     obj_attrib = nuim.get("attribute", attrib_token)
                bbox = [obj["bbox"][0], obj["bbox"][1], obj["bbox"][2] - obj["bbox"][0], obj["bbox"][3] - obj["bbox"][1]]
                # cv2.rectangle(img, (bbox[0], bbox[1]), (obj["bbox"][0] + bbox[2], obj["bbox"][1] + bbox[3]), (0, 255, 0), 1)
                entry.objects.append(Object(
                        obj_class=obj_class,
                        box2d=bbox,
                        box3d=None,
                        box3d_valid=False,
                        truncated=None,
                        occluded=None,
                    ))

        # img = cv2.resize(img, (800, 450))
        # cv2.imshow("NuImage", img)
        # cv2.waitKey(0)

        # upload to mongodb
        collection.insert_one(entry.get_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload 2D and 3D data from nuscenes dataset")
    parser.add_argument("--path", type=str, help="Path to nuscenes data, should contain samples/CAMERA_FRONT/*.jpg and v1.0-trainval/*.json folder e.g. /path/to/nuscenes")
    parser.add_argument("--version", type=str, help="NuImage version e.g. v1.0-train, v1.0-val, v1.0-mini", default="v1.0-train")
    parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
    parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
    parser.add_argument("--collection", type=str, default="nuimages_test", help="MongoDB collection")

    main(parser.parse_args())
