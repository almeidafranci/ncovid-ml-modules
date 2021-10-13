import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import datetime
from tensorflow.keras.models import load_model

import data_manner

def initial_data_processing(data_for_processing):
    data_processed = data_for_processing.copy()
    for column in data_processed.columns:
        data_processed[column] = data_processed[column].diff(7).dropna()
    
    data_processed = data_processed.dropna()
 
    data_final = [data_processed[col].values for col in data_processed.columns]

    return data_final

def predict_for_rquest(responsed_data):

    data = initial_data_processing(responsed_data)

    week_size = 7
    data_obj = data_manner.DataConstructor(week_size, is_training=False, type_norm=None)
    data_to_predict = data_obj.build_test(data)

    model_loaded = load_model('models/model_test_to_load')
    prediction = model_loaded.predict(data_to_predict.x, verbose=0)

    return prediction.reshape(-1)

def convert_to_string_format(output_of_prediction, begin, end):

    time_interval = pd.date_range(begin, end)

    returned_dictionaty = []
    for date, value in zip(time_interval, output_of_prediction):
        str_value = str(value)
        str_date = datetime.datetime.strftime(date, "%Y-%m-%d")
        returned_dictionaty.append({"date": str_date, "deaths": str_value})

    returned_str = str(returned_dictionaty)
    return returned_str

def time_delay(begin):
    start_date = datetime.datetime.strptime(begin, "%Y-%m-%d") - datetime.timedelta(days=7)
    return start_date.strftime("%Y-%m-%d")

def time_skip_for_predict(begin, end):
    start_date = datetime.datetime.strptime(begin, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    days_num = (end_date-start_date+datetime.timedelta(days=1)).days
    days_offset = days_num + (7 - days_num % 7)
    end_date = end_date + datetime.timedelta(days=days_offset-days_num)
    return end_date.strftime("%Y-%m-%d")

def predict(repo, path, feature, begin, end):
    original_end = end
    original_begin = begin
    
    # Always call before time_delay
    end = time_skip_for_predict(begin, end)
    begin = time_delay(begin)
    
    requested_data = pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/{repo}/path/{path}/feature/{feature}/begin/{begin}/end/{end}/as-csv", index_col='date')

    predicted_values = predict_for_rquest(requested_data)

    predictied_json = convert_to_string_format(predicted_values, original_begin, original_end)

    return predictied_json