import importlib_resources as resources
import yaml
import json
import jinja2
from jinja2 import Template
from pathlib import Path

from ginjinn import data_files
from ginjinn import config
from ginjinn.core import Configuration
from ginjinn.core.tf_dataset import TFDataset

''' Default configuration for ginjinn Project object. '''
DEFAULTS = Configuration({
    'annotation_path': 'ENTER PATH HERE',
    'annotation_type': 'PascalVOC',
    'image_dir': 'ENTER PATH HERE',
    'test_fraction': 0.25,
    'model_name': 'faster_rcnn_inception_v2_coco',
    'use_checkpoint': True,
    'checkpoint_path': '',
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

class MalformedConfigurationError(Exception):
    ''' Error indicating the project ist not yet set up '''
    pass

class ProjectNotReadyError(Exception):
    ''' Error indicating the project ist not yet set up '''
    pass

class Project:
    ''' A ginjinn Project object.

    The main object used to store configuration and run training,
    evalution, export and inference.
    '''
    def __init__(self, project_dir):
        project_path = Path(project_dir).resolve()

        self.config = Configuration({
            'project_dir': str(project_path),
            'dataset_dir': str(project_path.joinpath('dataset').resolve()),
            'model_dir': str(project_path.joinpath('model').resolve()),
            'export_dir': str(project_path.joinpath('export').resolve()),
            'config_path': str(project_path.joinpath('config.yaml').resolve()),
            'project_json': str(project_path.joinpath('project.json').resolve()),
            'ready': False,
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
        '''
            Write user-facing configuration yaml file
        '''
        template = Template(resources.read_text(data_files, 'config_template_jinja2.yaml'))
        rendered_template = template.render(config=self.config)
        with open(self.config.config_path, 'w') as f:
            f.write(rendered_template)

    def load_config(self):
        '''
            Update configuration from user-facing configuration yaml file
        '''
        with open(self.config.config_path) as f:
            try:
                _config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                msg = 'Could not parse config.yaml:\n{e}'
                raise MalformedConfigurationError(msg)
        
        self.config.update(_config)

    def to_json(self, fpath=None):
        '''
            Write internal configuration json file
        '''
        fpath = fpath or self.config.project_json
        with open(fpath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_json(self, fpath=None):
        '''
            Replace/Update configuration with configuration from json file.
        '''
        fpath = fpath or self.config.project_json
        with open(fpath) as f:
            # self.config = Configuration(json.load(f))
            self.config.update(Configuration(json.load(f)))

    def setup_project_dir(self):
        ''' Setup project directory and generate config files. '''
        # TODO: Add some info for user
        # create project directory if it does not exist
        Path(self.config.project_dir).mkdir(exist_ok=False)

        # generate user-facing config
        self.write_config()

        # generate internal config and update ready status
        self.config.ready = True
        self.to_json()
    
    def cleanup_project_dir(self):
        ''' Remove project directory '''
        
        # cleanup dataset dir
        self.cleanup_dataset_dir()

        # remove project files
        for fpath in self._project_files:
            path = Path(fpath)
            if path.exists():
                path.unlink()
        
        # remove project directory
        path = Path(self.config.project_dir)
        if path.exists():
            try:
                path.rmdir()
            except:
                msg = f'''Something went wrong cleaning up the project directory.
                Please remove directory "{str(path.resolve())}" manually.'''
                raise Exception()

        self.config.ready = False
    
    def cleanup_dataset_dir(self):
        ''' Remove project directory '''
        # cleanup dataset dir
        dataset = self._load_dataset()
        if dataset:
            dataset.cleanup_dataset_dir()

    # ==
    # Dataset
    # ==
    def is_ready(self):
        # TODO: Maybe check whether configuration files exist instead of storing ready state in config?
        #       Might be more robust
        return self.config.ready
    
    def setup_dataset(self):
        ''' Prepare input files for Tensorflow. This builds a dataset directory. '''
        # check if project is set up
        self._assert_project_is_ready()

        dataset = TFDataset(self.config.dataset_dir)
        dataset.construct_dataset(
            self.config.annotation_path,
            self.config.image_dir,
            self.config.annotation_type,
            self.config.test_fraction,
        )

    def is_ready_dataset(self):
        # check if project is set up
        self._assert_project_is_ready()

        dataset = self._load_dataset()
        if dataset:
            return dataset.is_ready()
        return False
    # ==

    @classmethod
    def from_directory(cls, project_dir):
        ''' Load Project object from directory. '''
        # TODO: print nicely formatted error
        Path(project_dir).resolve(strict=True)

        project = cls(project_dir)
        # load internal config
        project.load_json()
        # load potentially manipulated user-facing config
        # and update project config accordingly
        project.load_config()

        return project

    def _assert_project_is_ready(self):
        if not self.is_ready():
            raise ProjectNotReadyError(
                'Project directory is not yet set up. Run Project.setup_project_dir first.'
            )
    
    def _load_dataset(self):
        try:
            return TFDataset.from_directory(self.config.dataset_dir)
        except:
            return None

    @property
    def _project_files(self):
        return [
            self.config.config_path,
            self.config.project_json,
        ]

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

