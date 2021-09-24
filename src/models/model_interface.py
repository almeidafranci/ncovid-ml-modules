from math import sqrt

from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

import src.configuration as configs


class ModelInterface:
    def __init__(self, n_inputs, n_nodes, n_features, dropout=0.0, n_outputs=None):
        self.config = [str(n_inputs), str(n_nodes), str(n_features), str(dropout)]

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_features = n_features
        self.dropout = dropout
        self.n_outputs = None
        self.predictions = None
        self.stop_training = EarlyStopping(monitor='loss', mode='min', verbose=0,
                                           patience=configs.model_patience_earlystop)
        if n_outputs is None:
            self.n_outputs = n_inputs
        self.model = None

    def fit_model(self, x, y, verbose=0):
        self.model.fit(x=x,
                       y=y,
                       epochs=configs.model_train_epochs,
                       batch_size=configs.model_batch_size,
                       verbose=verbose,
                       callbacks=[self.stop_training])
        return self.model

    def make_predictions(self, data):
        """
        Make model predictions over data
        :param data: data to make predictions
        :return: prediction and prediction's rmse
        """
        yhat = self.model.predict(data.x, verbose=0)

        rmse_scores = list()
        for i in range(len(yhat)):
            mse = mean_squared_error(data.y[i], yhat[i])
            rmse = sqrt(mse)
            rmse_scores.append(rmse)

        return yhat, rmse_scores
