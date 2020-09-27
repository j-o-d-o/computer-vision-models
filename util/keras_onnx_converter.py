import keras2onnx
import onnx
import tensorflow as tf

H5_PATH = "/home/jodo/trained_models/kitti_mobile_ssd_22-12-2019-16-16-45/keras_model_0.h5"
ONNX_PATH = "/home/jodo/trained_models/kitti_mobile_ssd_22-12-2019-16-16-45/model.onnx"


if __name__ == "__main__":
  model = tf.keras.models.load_model(H5_PATH, compile=False)

  onnx_model = keras2onnx.convert_keras(model, "od_model", debug_mode=True)
  onnx.checker.check_model(onnx_model)

  onnx.save_model(onnx_model, ONNX_PATH)
