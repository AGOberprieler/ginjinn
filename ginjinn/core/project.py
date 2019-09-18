import importlib_resources as resources
import yaml
import jinja2
from jinja2 import Template
from pathlib import Path

from ginjinn import data_files
from ginjinn import config
from ginjinn.core import Configuration

''' Default configuration for ginjinn Project object. '''
DEFAULTS = Configuration({
    'annotation_path': None,
    'annotation_type': 'PascalVOC',
    'image_dir': None,
    'test_fraction': 0.25,
    'model_name': 'faster_rcnn_inception_v2_coco',
    'use_checkpoint': True,
    'checkpoint_path': None,
    'n_iter': 5000,
    'batch_size': 1,
    'augmentation': {
        'flip_horizontal': {
            'active': True,
        },
        'flip_vertical': {
            'active': True,
        },
        'flip_90': {
            'active': True,
        },
        'change_brightness': {
            'active': False,
            'min_delta': 0.1,
            'max_delta': 0.2,
        },
        'change_contrast': {
            'active': False,
            'min_delta': 0.8,
            'max_delta': 1.25,
        },
        'jitter_boxes': {
            'active': False,
            'ratio': 0.05,
        },
    }
})

class Project:
    ''' A ginjinn Project object.

    The main object used to store configuration and run training,
    evalution, export and inference.
    '''
    def __init__(self, project_dir):
        self.config = Configuration({
            'project_dir': str(Path(project_dir).resolve()),
            'data_dir': str(Path(project_dir).joinpath('data').resolve()),
            'model_dir': str(Path(project_dir).joinpath('model').resolve()),
            'export_dir': str(Path(project_dir).joinpath('export').resolve()),
            'config_path': str(Path(project_dir).joinpath('config.yaml').resolve()),
            'project_json': str(Path(project_dir).joinpath('project.json').resolve()),
            'annotation_path': DEFAULTS.annotation_path,
            'annotation_type': DEFAULTS.annotation_type,
            'image_dir': DEFAULTS.image_dir,
            'test_fraction': DEFAULTS.test_fraction,
            'model_name': DEFAULTS.model_name,
            'use_checkpoint': DEFAULTS.use_checkpoint,
            'checkpoint_path': DEFAULTS.checkpoint_path,
            'n_iter': DEFAULTS.n_iter,
            'batch_size': DEFAULTS.batch_size,
            'augmentation': {
                'flip_horizontal': {
                    'active': DEFAULTS.augmentation.flip_horizontal.active,
                },
                'flip_vertical': {
                    'active': DEFAULTS.augmentation.flip_vertical.active,
                },
                'flip_90': {
                    'active': DEFAULTS.augmentation.flip_90.active,
                },
                'change_brightness': {
                    'active': DEFAULTS.augmentation.change_brightness.active,
                    'min_delta': DEFAULTS.augmentation.change_brightness.min_delta,
                    'max_delta': DEFAULTS.augmentation.change_brightness.max_delta,
                },
                'change_contrast': {
                    'active': DEFAULTS.augmentation.change_contrast.active,
                    'min_delta': DEFAULTS.augmentation.change_contrast.min_delta,
                    'max_delta': DEFAULTS.augmentation.change_contrast.max_delta,
                },
                'jitter_boxes': {
                    'active': DEFAULTS.augmentation.jitter_boxes.active,
                    'ratio': DEFAULTS.augmentation.jitter_boxes.ratio,
                },
            },
        })

    def write_config(self):
        template = Template(resources.read_text(data_files, 'config_template_jinja.yaml'))
        rendered_template = template.render(config=self.config)
        with open(self.config.config_path, 'w') as f:
            f.write(rendered_template)


    # @staticmethod
    # def create_config_template(fpath):
    #     '''
    #     Create a configuration template file at fpath.

    #     Keyword arguments:
    #     fpath -- path to new config file
    #     '''
    #     template = resources.read_text(data_files, 'config_template.yaml')
    #     with open(fpath, 'w') as f:
    #         f.write(template)
