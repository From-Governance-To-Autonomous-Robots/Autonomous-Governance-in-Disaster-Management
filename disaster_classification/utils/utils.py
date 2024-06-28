import yaml
import os

def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, file_path='config.yaml'):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

def update_config(section, key, value, file_path='config.yaml'):
    config = load_config(file_path)
    if section in config:
        config[section][key] = value
    else:
        config[section] = {key: value}
    save_config(config, file_path)