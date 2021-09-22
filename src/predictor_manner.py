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
    data_obj = data_manner.Data(week_size, complete=False)
    data_to_predict = data_obj.build_test(data, type_norm=None)

    model_loaded = load_model('models/model_test_to_load')
    prediction = model_loaded.predict(data_to_predict.x, verbose=0)

    return prediction.reshape(-1)

def convert_output_to_json(output_of_prediction, begin, end):
    returned_dictionaty = []
    start_date = datetime.datetime.strptime(begin, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    date_interval = [start_date + datetime.timedelta(days=interval+7) for interval in range(0, (end_date-start_date).days+1)]
    string_date = []
    for date in date_interval:
        string_date.append(date.strftime("%Y-%m-%d"))

    for date, value in zip(string_date, output_of_prediction):
        returned_dictionaty.append({"date": date, "deaths": str(value)})

    returned_json = json.dumps(str(returned_dictionaty), indent=3, separators=(",", ":"))
    return returned_json

def time_delay(begin):
    start_date = datetime.datetime.strptime(begin, "%Y-%m-%d") - datetime.timedelta(days=7)
    return start_date.strftime("%Y-%m-%d")

def predict(repo, path, feature, begin, end):
 
    begin = time_delay(begin)
    
    requested_data = pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/{repo}/path/{path}/feature/{feature}/begin/{begin}/end/{end}/as-csv", index_col='date')

    predicted_values = predict_for_rquest(requested_data)

    predictied_json = convert_output_to_json(predicted_values, begin, end)

    return predictied_json