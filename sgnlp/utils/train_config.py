import json


def load_train_config(config_class, json_file_path):
    with open(json_file_path, "r") as f:
        json_file = json.load(f)
    return config_class(**json_file)
