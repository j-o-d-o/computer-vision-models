from tensorflow.keras.models import Model


class ResultSchema:
    """
    Container class for the results of the model
    """
    def __init__(self):
        self.metrics = {
            "training": {},
            "validation": {}
        }
        self.model = None
        self.max_batches_per_epoch: int = None
        self.max_epochs: int = None
        self.curr_epoch: int = -1  # -1 represents initialization
        self.curr_batch: int = 0

    def append_to_metric(self, metric_name: str, value: any, phase: str="training", epoch: int=None, batch: int=None):
        """
        Append a value to a specific metric
        :param metric_name: name of the metric
        :param value: value of the metric
        :param phase: can be "training" or "validation"
        :param epoch: integer to specify the epoch
        :param batch: integer to specify the batch
        """
        if phase not in self.metrics:
            raise ValueError("phase must be any of " + str(self.metrics.keys()))
        if epoch is None:
            epoch = self.curr_epoch
        if batch is None:
            batch = self.curr_batch
        if metric_name not in self.metrics[phase]:
            self.metrics[phase][metric_name] = []
        self.metrics[phase][metric_name].append({"value": float(value), "epoch": epoch, "batch": batch})

    def update_weights(self, model: Model, curr_epoch: int=None, curr_batch: int=None):
        """
        Update weights by updating the model member variable
        :param model: keras model
        :param curr_epoch: integer to specify the current epoch
        :param curr_batch: integer to specify the current batch
        :return:
        """
        if curr_epoch is not None:
            self.curr_epoch = curr_epoch
        if curr_batch is not None:
            self.curr_batch = curr_batch
        self.model = model

    def get_dict(self) -> dict:
        """
        Return the serialized class data as dict
        :return: class data as dict
        """
        return {
            "curr_epoch": self.curr_epoch,
            "curr_batch": self.curr_batch,
            "max_batches_per_epoch": self.max_batches_per_epoch,
            "max_epochs": self.max_epochs,
            "metrics": self.metrics
        }
