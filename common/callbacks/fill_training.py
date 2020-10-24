from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from common.schemas import TrainingSchema
from common.schemas.result import ResultSchema


class FillTraining(callbacks.Callback):
    """
    Base Class to fill training and result schemas
    """
    def __init__(
            self,
            name: str,
            keras_model: Model,
            epochs: int = None,
            batches_per_epoch: int = None):
        """
        :param name: name of the training as string
        :param keras_model: keras model that should be saved to the training
        :param epochs: integer of how many epochs are trained
        :param: batches_per_epoch: integer of how many batches per epoch are trained
        """
        self._keras_model = keras_model
        self._training = TrainingSchema(name, keras_model.get_config())
        self._training.result = ResultSchema()
        self._training.result.max_epochs = epochs
        self._training.result.max_batches_per_epoch = batches_per_epoch
        self._training.result.status = -1  # status for setup
        self.id = None

    def on_train_begin(self, logs=None):
        self._training.result.update_weights(self._keras_model)
        self._training.status = 100  # status for training

    def on_epoch_begin(self, epoch, logs=None):
        self._training.result.curr_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self._training.result.curr_batch = batch
        for key, value in logs.items():
            if key not in ["batch", "size"]:
                self._training.result.append_to_metric(key, value, phase="training")

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key.startswith('val_'):
                self._training.result.append_to_metric(key, value, phase="validation")
        self._training.result.update_weights(self._keras_model)

    def on_train_end(self, logs=None):
        self._training.status = 1  # status for training done
