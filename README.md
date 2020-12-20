# Computer Vision Models
A set of tensorflow models for different computer vision tasks.

## Getting started
#### External dependencies
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for package managment
- [MongoDB](https://docs.mongodb.com/manual/installation/) to store and read training data
#### Python dependencies
```bash
conda env create environmental.yml
conda activate computer-vision-models
```
Note: Tensorflow is installed via conda and should take care of compatible cuda and cudnn versions on its own and will not interfere with your current setup. The cost of this, is a delay in getting the latest tensorflow versions.
#### EdgeTpu support
In order to compile with EdgeTpu some tools need to be installed (https://coral.ai/docs/edgetpu/compiler)
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
# For compiler for TFLite models -> EdgeTpu
sudo apt-get install edgetpu-compiler
# For inference on EdgeTpu with python
sudo apt-get install python3-pycoral
```
In some cases there might be issues with accessing the EdgeTpu via USB without sudo (https://github.com/tensorflow/tensorflow/issues/32743)
```bash
# add user to plugdev to communicate with edge tpu without needing sudo
sudo usermod -aG plugdev $USER
# add this to /etc/udev/rules.d/99-edgetpu-accelerator.rules (might have to create the file first)
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",GROUP="plugdev"
SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",GROUP="plugdev"
```

## Structure
Overview of the folder structure and it's contents. A README.md in each folder provides more detailed documentation.
#### common
Includes all the common code shared between all the different model implementations. E.g. data reading, storing/plotting results, logging, saving models or converting models to different platforms.
#### data
In an attempt to be able to combine same kinds of data (e.g. semseg, 2D od, etc.) from different sources, custom label specs are used. Since data can also come in many different forms, all data is combined into MongoDB for easy storage and access. For each datasource an "upload script" exists that converts the source data to the internal label spec and uploads it do MongoDB.
#### models
Different model implementations for different computer vision tasks. Includes all necesarry pre- and post-processing, training, model description, inference, etc.
#### eval
Does not exist yet, work in progress. But the idea is that just like different data sources are combined to an internal label spec, different models implementing the same type of computer vision algo (e.g. semseg or 2D detections) should also output a common output spec to be evaluated against each other.

## Tests
Tests should be in the same location as the file that is tested with the naming convention $(FILE_NAME)_test.py. To run tests call `$ pytest` in the root directory or use your favorite test runner (e.g. pycharm or vs code).
