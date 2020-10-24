from tensorflow.keras.models import Model
from common.schemas.result import ResultSchema
from common.utils import Logger


class TrainingSchema:
    """
    Data Container for a Training
    """
    def __init__(self,
                 name: str,
                 keras_model: Model):
        """
        :param name: name of the training as string
        :param keras_model: keras model which is used to save its config and weights
        """
        # model info
        self.keras_model = keras_model
        self.name: str = name
        self.result: ResultSchema = None
        # training info
        self.status: int = 0
        self.log = ""

    def get_dict(self) -> dict:
        """
        :return: dict of serialized training data without weights
        """
        return_dict = {
            "name": self.name,
            "keras_model": self.keras_model,
            "status": self.status,
            "log": Logger.get_contents(),
            "curr_epoch": None,
            "curr_batch": None,
            "max_batches_per_epoch": None,
            "max_epochs": None,
            "metrics": None,
        }
        if self.result is not None:
            return_dict.update({
                "curr_epoch": self.result.curr_epoch,
                "curr_batch": self.result.curr_batch,
                "max_batches_per_epoch": self.result.max_batches_per_epoch,
                "max_epochs": self.result.max_epochs,
                "metrics": self.result.metrics
            })
        return return_dict
