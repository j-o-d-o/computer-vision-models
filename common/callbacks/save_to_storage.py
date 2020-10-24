import os
import sys
import json
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from common.callbacks.fill_training import FillTraining


class SaveToStorage(FillTraining):
  """
  Callback to save Trainings & Results to a local storage
  """
  def __init__(
      self,
      storage_path: str,
      name: str,
      keras_model: Model,
      save_initial_weights: bool = True):
    """
    :param storage_path: path to directory were the data should be stored
    :param name: name of the training as string
    :param keras_model: keras model that should be saved to the training
    :param save_initial_weights: boolean to determine if weights should be saved initally before training,
                                 default = True
    """
    super().__init__(name, keras_model)
    self._storage_path = storage_path + "/" + name + "_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    if not os.path.exists(self._storage_path):
      os.makedirs(self._storage_path)

    self._save_initial_weights = save_initial_weights
    self._keras_model = keras_model

  def on_train_begin(self, logs=None):
    super().on_train_begin(logs)
    if self._save_initial_weights:
      self.save()

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch, logs)
    # TODO: Check if model is better now and only save weights in that case
    save_tf = True
    self.save(save_tf)

  def save(self, save_tf: bool = True):
    epoch = self._training.result.curr_epoch

    if epoch <= 0:
      stdout_origin = sys.stdout
      sys.stdout = open(self._storage_path + "/network_architecture.txt", "w")
      try:
        self._keras_model.summary()
      except ValueError:
        pass
      sys.stdout.close()
      sys.stdout = stdout_origin

    data_dict = self._training.get_dict()
    json.dump(data_dict["metrics"], open(self._storage_path + "/metrics.json", 'w'))

    if save_tf:
      if epoch < 0:
        epoch = "init"

      # Save Model as h5 keras model to continue training
      # keras_file_name = self._storage_path + "/keras_model_" + str(epoch) + ".h5"
      # self._keras_model.save(keras_file_name, save_format="h5")

      # SavedModel for Tensorflow v2, not supported yet by ONNX and has some issues with the loading with custom objects
      # but once these issues are resolved this could be used for both, loading and continue training
      tf_export_dir = self._storage_path + "/tf_model_" + str(epoch)
      os.makedirs(tf_export_dir)
      tf.saved_model.save(self._keras_model, tf_export_dir)
