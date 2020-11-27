## Data upload to MongoDB
Scripts to upload different data sources to MongoDB with a unified data spec. All examples below expect a local running MongoDB.
In case you want to upload to a different MongoDB or adjust the database and collection, use the parameters. Check out the `--help` for
default parameters.

### semseg_spec.py & od_spec.py
Data specification for semseg and object detection. All data from the different sources is transformed and converted to
fit these specs. These enables to train the same model from different data sources without having to write specific processors for
each data source.</br>
While the semseg spec is basically the comma10k label spec, the od spec is close to the kitti label spec with minor additions (3D bounding box)
and minor changes (axis of coordinate system to autosar, different ignore area approach)

### Comma10k - Semseg
Semantic segmentation data from comma.ai (https://github.com/commaai/comma10k). Clone the repository to your machine and run
`>> comma10k.py --src_path /path/to/comma10k_repo`.

### Kitti - 2D and 3D Object Detection
Kitti contains about 7500 training images including 2D boxes and 3D data. Download from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d.
You will need:
- Download left color images of object data set (12 GB) (data_object_image_2.zip)
- Download training labels of object data set (5 MB) (data_object_label_2.zip)
- Download camera calibration matrices of object data set (16 MB) (data_object_calib.zip)

```bash
# Run script to convert Kitti data to od_spec and upload to MongoDB
kitti_3d.py --image_path /path/to/data_object_image_2/training/image_2 
            --label_path /path/to/data_object_label_2/training/label_2
            --calib_path /path/to/data_object_calib/training/calib
```

### Nuscenes - 2D and 3D Object Detection
Work in progress...<br>
Download: https://www.nuscenes.org/download (You will need to register and login first)
Dev-kit: https://github.com/nutonomy/nuscenes-devkit.git
