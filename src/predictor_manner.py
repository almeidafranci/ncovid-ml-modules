import pandas as pd
import datetime
from tensorflow.keras.models import load_model

import data_manner
import configs_manner

class PredictorConstructor:
    def __init__(self, repository, path, feature, begin, end):
        self.repo = repository
        self.path = path
        self.feature = feature
        self.begin = begin
        self.end = end
        self.window_size = configs_manner.model_window_size
        self.end_date = self.time_skip_for_predict()
        self.begin_date = self.time_delay()
        self.data = self.read_data()

    def time_skip_for_predict(self):
        start_date = datetime.datetime.strptime(self.begin, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(self.end, "%Y-%m-%d")
        days_num = (end_date-start_date+datetime.timedelta(days=1)).days
        days_offset = days_num + (self.window_size - days_num % self.window_size)
        end_date = end_date + datetime.timedelta(days=days_offset-days_num+self.window_size)
        return end_date.strftime("%Y-%m-%d")

    def time_delay(self):
        start_date = datetime.datetime.strptime(self.begin, "%Y-%m-%d") - datetime.timedelta(days=self.window_size)
        return start_date.strftime("%Y-%m-%d")

    def read_data(self):
        return pd.read_csv(f"http://ncovid.natalnet.br/datamanager/repo/{self.repo}/path/{self.path}/feature/{self.feature}/begin/{self.begin_date}/end/{self.end_date}/as-csv", index_col='date')
    
    def data_processing(self):
        data_copy = self.data
        for column in data_copy.columns:
            data_copy[column] = data_copy[column].diff(7).dropna()
        
        data_copy = data_copy.dropna()
        self.data_processed = [data_copy[col].values for col in data_copy.columns]

    def data_creator(self):
        self.data_obj = data_manner.DataConstructor(self.window_size, is_training=False, type_norm=None)
        self.data_to_predict = self.data_obj.build_test(self.data_processed)
    
    def model_load(self):
        self.model_loaded = load_model(configs_manner.model_path + 
                                        self.repo + 
                                        '_' + str(configs_manner.model_type) + 
                                        '_' + str(configs_manner.model_window_size) + 
                                        '_' + str(configs_manner.model_train_features) + 
                                        '_' + str(configs_manner.model_neurons) + '.h5')
    
    def predict_for_data(self):
        self.predicted = self.model_loaded.predict(self.data_to_predict.x, verbose=0).reshape(-1)
    
    def convert_to_string_format(self):
        time_interval = pd.date_range(self.begin, self.end)

        returned_dictionaty = []
        for date, value in zip(time_interval, self.predicted):
            str_value = str(value)
            str_date = datetime.datetime.strftime(date, "%Y-%m-%d")
            returned_dictionaty.append({"date": str_date, "deaths": str_value})

        self.json_response = str(returned_dictionaty)
    
    def predict(self):
        self.data_processing()
        self.data_creator()
        self.model_load()
        self.predict_for_data()
        self.convert_to_string_format()

        return self.json_response