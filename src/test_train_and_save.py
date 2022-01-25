import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import data_manner
from models import lstm_manner
import configs_manner as configs

# Requesting the data.
raw_data = pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/p971074907/path/brl:rn/feature/date:deaths:newCases:/begin/2020-05-01/end/2021-07-01/as-csv", index_col='date')

# Data length
data_length = raw_data.shape[0]

# Setting the week size.
week_size = configs.model_window_size

# Condition to know if the raw data length is a week_size multiple.
if data_length%week_size !=0:
   print("Dataset lenght " + str(data_length) + " is not divisible by the week size.")
   data_length_plus_offset = data_length + (week_size - data_length % week_size)
   print("Add more " + str(data_length_plus_offset-data_length) + " days to request.")

else:
   # If you want to use diff().
   # Turning newCases in an accumulated cases data.
   raw_data['cases'] = raw_data['newCases'].cumsum()
   data_to_processed = raw_data[['deaths', 'cases']].copy()

   # Applying the diff() function.
   for column in data_to_processed.columns:
      data_to_processed[column] = data_to_processed[column].diff(7).dropna()

   data_to_processed = data_to_processed.dropna()
   data_to_construct = [data_to_processed[col].values for col in data_to_processed.columns]

   test_size = 42
   #First create a train and test data
   data_to_train_obj = data_manner.DataConstructor(week_size, is_training=True, type_norm=None)
   train, _ = data_to_train_obj.build_train_test(data_to_construct, test_size)

   #Data for test, just because I want all data to plot
   data_to_validation_obj = data_manner.DataConstructor(week_size, is_training=False, type_norm=None)
   test = data_to_validation_obj.build_test(data_to_construct)

   # Train
   layer_size = 250
   dropout = 0.15

   # features [ deaths and cases]
   features_qtd = 2

   model_to_train = lstm_manner.ModelLSTM(n_inputs=week_size, n_nodes=layer_size, dropout=dropout, n_features=features_qtd)

   model_to_train.fit_model(train.x, train.y)

   # Test all data
   yhat, rmse_scores = model_to_train.make_predictions(test)

   yhat_flatten = yhat.flatten()

   # se quiser tire
   plt.title("Média dos RMSE total e apenas de teste " + str(round(np.mean(rmse_scores),3)) 
               + " || " + str(round(np.mean(rmse_scores[:-42]), 3)))
   plt.plot(data_to_construct[0][:-test_size])
   plt.plot(yhat_flatten)
   plt.show()

   #Save model
   #model_to_train.save_model('p971074907')