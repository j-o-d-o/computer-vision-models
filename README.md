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

## Structure
Overview of the folder structure and it's contents. A README.md in each folder provides more detailed documentation.
#### common
Includes all the common code shared between all the different model implementations. E.g. data reading, storing/plotting results, logging, saving models or converting models to different platforms.
#### data
In an attempt to be able to combine same kinds of data (e.g. semseg, 2D od, etc.) from different sources custom label specs are used. Since data can also come in many different forms, all data is combined into MongoDB for easy storage and access. Each data source used gets a "upload script" that converts the source data to the internal label spec and uploads it do MongoDB.
#### models
Different model implementations for different computer vision tasks. Includes all necesarry pre- and post-processing, training, model description, inference, etc.
#### eval
Does not exist yet, work in progress. But the idea is that just like different data sources are combined to an internal label spec, different models implementing the same type of computer vision algo (e.g. semseg or 2D detections) should also output a common output spec to be evaluated against each other.

## Tests
Tests should be in the same location as the file that is tested with the naming convention $(FILE_NAME)_test.py. To run tests call `$ pytest` in the root directory or use your favorite test runner (e.g. pycharm or vs code).
