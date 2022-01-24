import json


def load_config():
    # Opening JSON file
    with open('../doc/configure.json') as json_file:
        data = json.load(json_file)

        docs_path = data['folder_configs']['docs_path']
        data_path = data['folder_configs']['data_path']
        model_path = data['folder_configs']['data_path'] + data['folder_configs']['model_path']
        glossary_file = data['folder_configs']['docs_path'] + data['folder_configs']['glossary_file']

        model_batch_size = data['model_configs']['batch_size']
        model_epochs = data['model_configs']['epochs']
        model_patience_earlystop = data['model_configs']['patience_earlystop']
