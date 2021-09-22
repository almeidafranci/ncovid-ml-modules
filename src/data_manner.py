from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error

from model_manner import build_model

class Data:
    def __init__(self, step_size, complete=False):
        self.step_size = step_size
        self.complete = complete

    class Test:
        def __init__(self, data, type_norm=None):
            self.x = data[:, :, 1:]  # self.x = x
            self.y = data[:, :, :1]
            self.y = self.y.reshape((self.y.shape[0], self.y.shape[1], 1))
            self.type_norm = type_norm
            self._y_hat = None
            self._rmse = None

        @property
        def y_hat(self):
            return self._y_hat

        @y_hat.setter
        def y_hat(self, y_hat_list):
            self._y_hat = y_hat_list
            self._rmse = list()
            for i in range(len(self.y_hat)):
                mse_i = mean_squared_error(self.y[i], self.y_hat[i])
                rmse_i = sqrt(mse_i)
                self._rmse.append(rmse_i)

        @property
        def rmse(self):
            return self._rmse

        @rmse.setter
        def rmse(self, rmse_list):
            self._rmse = rmse_list


    class Train:
        def __init__(self, data, step_size, type_norm=None):
            x, y = self.to_supervised(data, step_size)
            self.x_labeled = x[:, :, : 1]
            self.x = x[:, :, 1:]  # self.x = x
            self.y = y.reshape((y.shape[0], y.shape[1], 1))
            self.type_norm = type_norm
            self._y_hat = None
            self._rmse = None

        @property
        def y_hat(self):
            return self._y_hat

        @y_hat.setter
        def y_hat(self, y_hat_list):
            self._y_hat = y_hat_list
            self._rmse = list()
            for i in range(len(self.y_hat)):
                mse_i = mean_squared_error(self.y[i], self.y_hat[i])
                rmse_i = sqrt(mse_i)
                self._rmse.append(rmse_i)

        @property
        def rmse(self):
            return self._rmse

        @rmse.setter
        def rmse(self, rmse_list):
            self._rmse = rmse_list

        @staticmethod
        def to_supervised(data_, n_input):
            # flatten data
            data = data_.reshape((data_.shape[0] * data_.shape[1], data_.shape[2]))
            x, y = list(), list()
            in_start = 0
            # step over the entire history one time step at a time
            for _ in range(len(data)):
                # define the end of the input sequence
                in_end = in_start + n_input
                out_end = in_end + n_input
                # ensure we have enough data for this instance
                if out_end < len(data):
                    x.append(data[in_start:in_end, :])
                    y.append(data[in_end:out_end, 0])
                # move along one time step
                in_start += 1
            return np.array(x), np.array(y)


    def build_data(self, data, size_data_test, type_norm=None):
        data = self.data_reshape(data)
        data_train, data_test = Data.split_train_test(data, size_data_test, self.step_size)
        train = self.build_train(data_train, type_norm)
        test = self.build_test(data_test, type_norm)
        return train, test

    def data_reshape(self, data):
        return np.array(data).T

    def janelamento(self, data):
        return np.array(np.split(data, len(data) / self.step_size))
    
    def build_train(self, data, type_norm):
        if not self.complete:
            data = self.data_reshape(data)
        data = self.janelamento(data)
        return self.Train(data, self.step_size, type_norm)

    def build_test(self, data, type_norm):
        if not self.complete:
            data = self.data_reshape(data)
        data = self.janelamento(data)
        return self.Test(data, type_norm)

    def split_train_test(self, data, n_test):
        # makes dataset multiple of n_days
        data = data[data.shape[0] % self.step_size:]
        # make test set multiple of n_days
        n_test -= n_test % self.step_size
        # split into standard weeks
        train, test = data[:-n_test], data[-n_test:]
        return train, test
