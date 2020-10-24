from bson import ObjectId
import gridfs
import os
from tensorflow.keras.models import Model
from typing import Tuple, Callable
from common.utils import Config
from common.data_reader.mongodb import MongoDBConnect
from common.callbacks.fill_training import FillTraining


class SaveToMongoDB(FillTraining):
    """
    Callback to save Trainings & Results to a local MongoDB
    """
    def __init__(
            self,
            db_config: Tuple[str, str],
            name: str,
            keras_model: Model,
            save_initial_weights: bool=True,
            epoch_save_condition: Callable=None):
        """
        :param db_config: Tuple of string which specify the database connection [Name in Config, Database name]
        :param name: name of the training as string
        :param keras_model: keras model that should be saved to the training
        :param save_initial_weights: boolean to determine if weights should be saved initally before training,
                                     default = True
        :param epoch_save_condition: a function to test if the weights of the model should be saved after the epoch,
                                     the function takes an TrainingSchema as argument. It defaults to None which
                                     will save the weights after each epoch
        """
        super().__init__(name, keras_model)
        self._epoch_save_condition = epoch_save_condition
        self._save_initial_weights = save_initial_weights
        self.id = None

        # connect to database
        self.mongo_con = MongoDBConnect()
        self.mongo_con.add_connections_from_config(Config.get_config_parser())
        self._db = self.mongo_con.get_db(*db_config)
        self._collection = self._db["training"]

        # save training on init
        self.save()

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.update()
        if self._save_initial_weights:
            self.update_weights()

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        self.update()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.update()
        if self._epoch_save_condition is None or self._epoch_save_condition(self._training):
            self.update_weights()

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.update()
        self.mongo_con.reset_connections()
        print("Saved Training to Database with ObjectId: " + str(self.id))

    def save(self):
        data_dict = self._training.get_dict()
        data_dict["metrics"] = None
        data_dict["weights"] = []
        self.id = self._collection.insert_one(data_dict).inserted_id

    def update(self):
        data_dict = self._training.get_dict()
        self._collection.update_one(
            {'_id': ObjectId(self.id)},
            {
                '$set': data_dict
            }
        )

    def update_weights(self):
        fs = gridfs.GridFS(self._collection.database)
        tmp_filename = "tmp_model_weights_save.h5"
        if self._training.result.model is not None:
            self._training.result.model.save(tmp_filename)
            with open(tmp_filename, mode='rb') as file:
                file_bytes = file.read()
                model_gridfs = fs.put(file_bytes)
            os.remove(tmp_filename)

            weights = {
                "model_gridfs": model_gridfs,
                "epoch": self._training.result.curr_epoch,
                "batch": self._training.result.curr_batch
            }

            self._collection.update_one(
                {'_id': ObjectId(self.id)},
                {'$push': {'weights': weights}}
            )
