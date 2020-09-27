import argparse
from nuscenes.scripts import export_kitti


def main(args):
  # TODO: adapt KittiConverter via PR to take dataroot as arg and to not expect every image to exist (https://github.com/nutonomy/nuscenes-devkit)
  converter = export_kitti.KittiConverter(
    dataroot=args.path,
    nusc_kitti_dir=args.path + "/kitti_format",
    image_count=-1,
    nusc_version="v1.0-trainval",
    split="train"
  )
  converter.nuscenes_gt_to_kitti()
  # TODO: upload as kitti data once conversion was successful


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Upload 2D and 3D data from nuscenes dataset")
  parser.add_argument("--path", type=str, help="Path to nuscenes data, should contain samples/CAMERA_FRONT/*.jpg and v1.0-trainval/*.json folder e.g. /path/to/nuscenes")
  parser.add_argument("--conn", type=str, default="mongodb://localhost:27017", help='MongoDB connection string')
  parser.add_argument("--db", type=str, default="object_detection", help="MongoDB database")
  parser.add_argument("--collection", type=str, default="kitti", help="MongoDB collection")

  main(parser.parse_args())
