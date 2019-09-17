import importlib_resources as resources
from ginjinn import data_files
from pathlib import Path

from ginjinn import config

class Project:
    def __init__(self):
        pass

    @staticmethod
    def create_config_template(fpath):
        '''
        Create a configuration template file at fpath.

        Keyword arguments:
        fpath -- path to new config file
        '''
        template = resources.read_text(data_files, 'config_template.yaml')
        with open(fpath, 'w') as f:
            f.write(template)
