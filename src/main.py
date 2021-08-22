import pandas as pd
import matplotlib.pyplot as plt

import data_manner
import model_manner


def read_csv_file(path, column_date, last_day, first_date=None):
    df = pd.read_csv(path)
    if first_date is not None:
        return df[(df[column_date] < last_day) & (df[column_date] >= first_date)]
    return df[(df[column_date] < last_day)][1:]


db_folder = '../dbs/'
data_limite = '2021-03-21'
df_araraquara = read_csv_file(db_folder + 'df_araraquara.csv', 'date', data_limite, None)
df_araraquara.confirmed = df_araraquara.confirmed.diff(7).dropna()
df_araraquara.deaths = df_araraquara.deaths.diff(7).dropna()
df_araraquara = df_araraquara.dropna()

# ---------------------

data = [df_araraquara.deaths.values,
        df_araraquara.confirmed.values]

week_size = 7
train, test = data_manner.build_data(data, step_size=week_size, size_data_test=21)

# model_config: [n_input, n_lstm_nodes, dropout, n_features]
model_config = [week_size, 200, 0.0, train.x.shape[2]]

# # ---------- Sample 1 - Without griding-search
# regressor = model_manner.build_model(model_config)
#
# regressor.fit_model(train, epochs=1, batch_size=16, verbose=0)
#
# train.y_hat, train.rmse = evaluator_manner.evaluate_model(regressor.model, train)
# test.y_hat, test.rmse = evaluator_manner.evaluate_model(regressor.model, test)
#
# print(sum(test.rmse))
#
# plt.rcParams["figure.figsize"] = (20, 5)
# plt.plot(train.x_label.reshape(train.x_label.shape[0], train.x_label.shape[1])[:, :1], label='original')
# plt.plot(train.y_hat.reshape(train.y_hat.shape[0], train.y_hat.shape[1])[:, :1], label='predicted')
# plt.show()

# ----------- Sample 2 - griding-search for a single configuration model based over the train data
grid_size = 3
grid_result_list = model_manner.grid_model(grid_size, model_config, train, epochs=10, batch_size=32, verbose=0)

# grid_result_list[i][0]: LSTMRegressor object
# grid_result_list[i][1]: y_hat list of the previous LSTMRegressor object
# grid_result_list[i][2]: rmse list of the previous LSTMRegressor object

for i in range(len(grid_result_list)):
    print(sum(grid_result_list[i][2]))

plt.rcParams["figure.figsize"] = (20, 5)
plt.plot(train.y.reshape(train.y.shape[0], train.y.shape[1])[:, :1], label='original')
plt.plot(grid_result_list[0][1].reshape(grid_result_list[0][1].shape[0], grid_result_list[0][1].shape[1])[:, :1], label='predicted')
plt.show()
