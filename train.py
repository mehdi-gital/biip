# reads data, loads models, loads configs, trains and creates model artifacts
import configparser
import os

project_name = 'experiment_0'

config = configparser.ConfigParser()
config.read(os.path.join('data', project_name, 'configs.ini'))
config.sections()