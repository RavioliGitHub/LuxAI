"""
Functions to read Config files.
"""

import yaml


# Load YAML configuration file
def read_yaml_file(yaml_file):
    with open(yaml_file, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config
