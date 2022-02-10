import numpy as np
from data_manner import Train, Test


class Evaluator:
    def __init__(self, model=None, data_train=None, data_test=None, n_repeat=1):
        self._data_train = data_train
        self._data_test = data_test
        self.n_repeat = n_repeat
        self._model = model
        self._models = list()
        if model is not None:
            self._models.append(model)

    @property
    def data_train(self):
        return self._data_train

    @data_train.setter
    def data_train(self, train):
        self._data_train = train

    @property
    def data_test(self):
        return self._data_test

    @data_test.setter
    def data_test(self, test):
        self._data_test = test

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_model):
        self._model = new_model
        self._models.append(new_model)

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, new_models):
        self._models = new_models

    def clean_models(self):
        self._models = list()

    def evaluate_model(self, model=None, data_train=None, data_test=None):
        """Evaluate model over train and test

        Args:
            model (model, optional): model trained. Defaults to None.
            data_train (Train, optional): Data train to be trained. Defaults to None.
            data_test (Test, optional): Data test to be evaluated. Defaults to None.

        Returns:
            y_hats, rmses: predictions and rmses
        """

        if data_train is None:
            data_train = self.data_train
        if data_test is None:
            data_test = self.data_test
        if model is None:
            model = self._model

        # walk-forward validation over each week
        history = data_train
        history.y_hat = list()
        history.rmse = list()
        y = list()
        for idx, num in enumerate(data_test.x):
            y.append(history.y)
            # predict the week
            yhat, rmse = model.predicting(history)
            # store the predictions
            history.y_hat.append(yhat)
            history.rmse.append(rmse)
            # get real observation and add to history for predicting the next week
            history.x = np.vstack((history.x, data_test.x[idx : idx + 1 :,]))
            history.y = np.vstack((history.y, data_test.y[idx : idx + 1 :,]))
        # evaluate predictions days for each week
        # predictions = np.array(predictions)
        y = y[-1].reshape(y[-1].shape[0], ys[-1].shape[1])[:, :1]
        history.y_hat = history.y_hat[-1].reshape(
            history.y_hat[-1].shape[0], history.y_hat[-1].shape[1]
        )[:, :1]
        return y, history.y_hat, history.rmse[-1]

    def evaluate_model_n_times(
        self, model=None, train=None, test=None, n_repeat=None, verbose=0
    ):
        """
        Fit and Evaluate a single model over train and test multiple times
        :param model: Specify model to training and evaluate
        :param train: Specify train temporal series to evaluate or use the train temporal series inserted in class
        :param test: Specify test temporal series to evaluate or use the test temporal series inserted in class
        :param n_repeat:
        :param verbose: Specify training should be verbose or silent
        :return: regressor_list with predictions and rmses for unique model
        """
        if train is None:
            train = self.data_train
        if test is None:
            test = self.data_test
        if model is None:
            model = self._model
        if n_repeat is None:
            n_repeat = self.n_repeat

        regressor_list = list()
        y_list = list()
        y_hat_list = list()
        rmse_list = list()
        for idx, num in enumerate(n_repeat):
            regressor_list.append(model)
            regressor_list[idx].fiting(train.x, train.y, verbose)
            y, y_hat, rmse = self.evaluate_model(model, train, test)
            y_list.append(y)
            y_hat_list.append(y_hat)
            rmse_list.append(rmse)

        return list(zip(regressor_list, y_list, y_hat_list, rmse_list))

    def evaluate_n_models_n_times(
        self, list_models=None, train=None, test=None, n_repeat=1, verbose=0
    ):
        """
        Fit and Evaluate multiple models over train and test multiple times
        :param list_models: Specify model list to training and evaluate
        :param train: Specify train temporal series to evaluate or use the train temporal series inserted in class
        :param test: Specify test temporal series to evaluate or use the test temporal series inserted in class
        :param verbose: Specify training should be verbose or silent
        :return: regressors_list with predictions and rmses for any model from list models
        """

        if list_models is None:
            list_models = self._models

        regressors_list = list()

        for model in list_models:
            regressors_list.append(
                self.evaluate_model_n_times(model, train, test, n_repeat, verbose)
            )
            self._model = model

        return regressors_list

    def __str__(self):
        if type(self._data_train) is Train and type(self._data_test) is Test:
            return (
                f"\nQuantity Models: {len(self._models)}"
                f"\nLast Model added and settled: {self._model}"
                f"\nData train: {self._data_train.x.shape}"
                f"\nData test: {self._data_test.x.shape}"
                f"\nRepetitions: {self.n_repeat}\n"
            )

        return (
            f"\nQuantity Models: {len(self._models)}"
            f"\nLast Model added and settled: {self._model}"
            f"\nData train: {self._data_train}"
            f"\nData test: {self._data_test}"
            f"\nRepetitions: {self.n_repeat}\n"
        )
