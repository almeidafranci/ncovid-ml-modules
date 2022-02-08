import logger
from math import sqrt
from sklearn.metrics import mean_squared_error

import configs_manner
from models.model_interface import ModelInterface
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf


class ModelArtificalInterface(ModelInterface):
    def __init__(self, locale):
        super().__init__(locale)
        self.nodes = configs_manner.model_infos["model_nodes"]
        self.epochs = configs_manner.model_infos["model_epochs"]
        self.dropout = configs_manner.model_infos["model_dropout"]
        self.batch_size = configs_manner.model_infos["model_batch_size"]
        self.is_output_in_input = configs_manner.model_infos["model_is_output_in_input"]
        self.is_predicting = configs_manner.model_is_predicting
        self.data_window_size = configs_manner.model_infos["data_window_size"]
        self.earlystop = EarlyStopping(
            monitor="loss",
            mode="min",
            verbose=0,
            patience=configs_manner.model_infos["model_earlystop"],
        )
        self.n_features = configs_manner.model_infos["data_n_features"]

    def _resolve_model_name(self, is_remote=False):
        return (
            str(self.model_path_remote if is_remote else self.model_path)
            + self.locale
            + "_"
            + self.model_subtype
            + "_"
            + str(self.data_window_size)
            + "_"
            + str(self.n_features)
            + "_"
            + str(self.nodes)
            + "_"
            + str(int(self.dropout * 100))
            + ".h5"
        )

    def saving(self):
        self.model.save(self._resolve_model_name())

    def loading(self, model_name=None):
        try:
            if model_name:
                return tf.keras.models.load_model(model_name)
            return tf.keras.models.load_model(self._resolve_model_name())
        except OSError:
            logger.debug_log(
                self.__class__.__name__, self.loading.__name__, "Model local failed"
            )
            # TO DO
            # load from web
        except Exception as e:
            logger.error_log(
                self.__class__.__name__,
                self.loading.__name__,
                "Load model - {}".format(e.__traceback__.__str__),
            )

    def fiting(self, x, y, verbose=0):
        try:
            self.model.fit(
                x=x,
                y=y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[self.earlystop],
                verbose=verbose,
            )
            logger.debug_log(
                self.__class__.__name__, self.fiting.__name__, "Model fitted"
            )
            return True
        except:
            logger.error_log(self.__class__.__name__, self.fiting.__name__, "Fit model")

    def predicting(self, data):
        """
        Make predictions (often test data)
        :param data: data to make predictions
        :return: prediction and prediction's rmse
        """
        yhat = self.model.predict(data.x, verbose=0)
        rmse_scores = list()
        for idx, _ in enumerate(yhat):
            mse = mean_squared_error(data.y[idx], yhat[idx])
            rmse = sqrt(mse)
            rmse_scores.append(rmse)

        return yhat, rmse_scores
