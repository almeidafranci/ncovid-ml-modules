#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:11:53 2022

@author: davi
"""

import json
from urllib.request import urlopen
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import data_manner
from models import lstm_manner
import configs_manner as configs

# load data from wcota
rd = pd.read_csv("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv", index_col='date')
 
# returns JSON object as a dictionary
response = urlopen("https://gist.githubusercontent.com/henriquejensen/1032c47a44d2cddaa2ef47fc531025db/raw/c58fdc848baf2a1fb53e617a0ad4e9754ec68e35/json-estados-brasileiros")
data = json.loads(response.read())
 
# for each state
for state in data['UF']:
  # get deaths and newCases columns
  raw_data = rd[rd['state']==state['sigla']][['deaths', 'newCases']]
  
  # Data length
  data_length = raw_data.shape[0]
  
  # Setting the week size.
  week_size = configs.model_window_size
  
  if data_length%week_size !=0:
    raw_data = raw_data[:-(data_length % week_size)]
  
  # Condition to know if the raw data length is a week_size multiple.
  # If you want to use diff().
  # Turning newCases in an accumulated cases data.
  raw_data['cases'] = raw_data['newCases'].cumsum()
  data_to_processed = raw_data[['deaths']].copy()
 
  # Applying the diff() function.
  for column in data_to_processed.columns:
     data_to_processed[column] = data_to_processed[column].diff(7).dropna()
 
  data_to_processed = data_to_processed.dropna()
  data_to_construct = [data_to_processed[col].values for col in data_to_processed.columns]
 
  test_size = 7
  #First create a train and test data
  data_to_train_obj = data_manner.DataConstructor(week_size, is_training=True, type_norm=None)
  train, _ = data_to_train_obj.build_train_test(data_to_construct, test_size)
 
  #Data for test, just because I want all data to plot
  data_to_validation_obj = data_manner.DataConstructor(week_size, is_training=False, type_norm=None)
  test = data_to_validation_obj.build_test(data_to_construct)
 
  # Train
  layer_size = 300
  dropout = 0.0
 
  # features [ deaths and cases]
  features_qtd = 1
 
  model_to_train = lstm_manner.ModelLSTM(n_inputs=week_size, n_nodes=layer_size, dropout=dropout, n_features=features_qtd)
 
  model_to_train.fit_model(train.x, train.y, verbose=1)
 
  # Test all data
  yhat, rmse_scores = model_to_train.make_predictions(test)
 
  yhat_flatten = yhat.flatten()
 
  # se quiser tire
  plt.title("Média dos RMSE total e apenas de teste " + str(round(np.mean(rmse_scores),1)) 
              + " || " + str(round(np.mean(rmse_scores[:-test_size]), 1)) + " para " + state['sigla'])
  plt.plot(data_to_construct[0][:-test_size])
  plt.plot(yhat_flatten)
  plt.show()
 
  #Save model
  model_to_train.save_model(state['sigla'])
  
  # start date
  # input
 
# Closing file
#f.close()

#print(raw_data[raw_data['state']=='TOTAL'])
#print(raw_data.columns)