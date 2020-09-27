## Some utility functions

### save_to_storage.py
Callback implementation for mlpipe to save a model to the local harddrive as a tensorflow model as well as save metrics as
json file.

### plot_metrics.py
Metrics that are saved by save_to_storage.py can be visualized in a graph.

### tflite_converter.py
Converting a tensorflow model to tflite model and optionally quantize the weights. For running the model on edgetpu
there is one more compile step with the edgetpu command line tool needed (https://coral.ai/docs/edgetpu/compiler):
```bash
# install edgetpu command line tool
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
# compile tflite model to tflite model for edgetpu
edgetpu-compiler path/to/model.tflite
```

### keras_onnx_convert.py
Converting a keras model to ONNX format. In order to use TensorRT on C++ side for inference the Tensorflow
SavedModel or Keras Model has to be converted to ONNX Format (https://github.com/onnx/onnx).

For Keras model just use the `keras_onnx_convert.py` script.

For Tensorflow SaveModel format use the command line: (currently has some issues, probably have to wait for Tensorflow 2 support)
```bash
>> conda activate object-detection
>> python -m tf2onnx.convert\
    --saved-model /home/jodo/trained_models/kitti_mobile_ssd_17-12-2019-10-42-43/tf_model_15\
    --output /home/jodo/trained_models/kitti_mobile_ssd_17-12-2019-10-42-43/tf_model_15/model.onnx\
    --verbose
```
