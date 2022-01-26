import json
import sys
import os.path

sys.path.append('../')

assert(os.path.isfile('../doc/configure.json')), 'Config file unreached'
with open('../doc/configure.json') as json_file:
    data = json.load(json_file)

    doc_folder = data['folder_configs']['docs_path']
    data_path = data['folder_configs']['data_path']
    model_path = data['folder_configs']['data_path'] + data['folder_configs']['model_path']
    glossary_file = data['folder_configs']['docs_path'] + data['folder_configs']['glossary_file']

    model_type = data['model_configs']['type']
    model_window_size = data['model_configs']['window_size']
    model_batch_size = data['model_configs']['batch_size']
    model_train_epochs = data['model_configs']['epochs']
    model_train_features = data['model_configs']['train_features']
    model_neurons = data['model_configs']['neurons']
    model_patience_earlystop = data['model_configs']['patience_earlystop']
    model_is_training = eval(data['model_configs']['is_training'])
    model_is_output_in_input = eval(data['model_configs']['is_output_in_input'])
