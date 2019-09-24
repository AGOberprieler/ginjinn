import importlib_resources as resources
import yaml
import json
import jinja2
from jinja2 import Template
from pathlib import Path
import shutil

from ginjinn import data_files
from ginjinn import config
from ginjinn.core import Configuration
from ginjinn.core.tf_dataset import TFDataset, DatasetNotReadyError
from ginjinn.core.tf_model import TFModel, ModelNotReadyError, ModelNotTrainedError, ModelNotExportedError
from ginjinn.core.tf_augmentation import TFAugmentation

''' Default configuration for ginjinn Project object. '''
DEFAULTS = Configuration({
    'annotation_path': 'ENTER PATH HERE',
    'annotation_type': 'PascalVOC',
    'image_dir': 'ENTER PATH HERE',
    'test_fraction': 0.25,
    'model': 'faster_rcnn_inception_v2_coco',
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
            'model': DEFAULTS.model,
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

        try:
            TFAugmentation(_config['augmentation'])
        except:
            msg = 'Could not parse augmentation options. Check your config.yaml'
            raise MalformedConfigurationError(msg)
        

        # resolve paths to allow '~' in path on linux
        _config['image_dir'] = str(Path(_config['image_dir']).resolve(strict=True))
        _config['annotation_path'] = str(Path(_config['annotation_path']).resolve(strict=True))
        if _config['checkpoint_path']:
            _config['checkpoint_path'] = str(Path(_config['checkpoint_path']).resolve(strict=True))

        print(_config)

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

    def setup_project_dir(self, force=False):
        ''' Setup project directory and generate config files. '''
        # TODO: Add some info for user
        # create project directory if it does not exist
        if Path(self.config.project_dir).exists() and force:
            shutil.rmtree(self.config.project_dir)
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

    def cleanup_data_model_export(self):
        self.cleanup_dataset_dir()
        self.cleanup_model_export()
        self.cleanup_model_training()
        self.cleanup_model_dir()

    def is_ready(self):
        # TODO: Maybe check whether configuration files exist instead of storing ready state in config?
        #       Might be more robust
        return self.config.ready
    # ==
    # Dataset
    # ==
    def setup_dataset(self, force=False):
        ''' Prepare input files for Tensorflow. This builds a dataset directory. '''
        # check if project is set up
        self._assert_project_is_ready()

        if Path(self.config.dataset_dir).exists() and not force:
            raise Exception('Dataset already exists. Rerun with --force if you want to overwrite it.')

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

    def cleanup_dataset_dir(self):
        ''' Remove project directory '''
        # cleanup dataset dir
        dataset = self._load_dataset()
        if dataset:
            dataset.cleanup_dataset_dir()
    
    def dataset_summary(self):
        self._assert_project_is_ready()
        self._assert_dataset_is_ready()
        
        dataset = self._load_dataset()
        return dataset.get_summary()
    # ==

    # ==
    # Model
    # ==
    def setup_model(self, force=False):
        self._assert_project_is_ready()

        if Path(self.config.model_dir).exists() and not force:
            raise Exception('Model already exists. Rerun with --force if you want to overwrite it.')
        
        dataset = self._load_dataset()
        if not dataset:
            raise DatasetNotReadyError('Dataset not ready. Run Project.setup_dataset first')

        model = TFModel(self.config.model_dir)
        model.construct_model(
            self.config.model,
            dataset.config.record_train_path,
            dataset.config.record_eval_path,
            dataset.config.labelmap_path,
            self.config.checkpoint_path,
            self.config.use_checkpoint,
            self.config.augmentation,
            self.config.n_iter,
            self.config.batch_size,
            self.config.augmentation,
        )

    def is_ready_model(self):
        # check if model is setup
        self._assert_project_is_ready()

        model = self._load_model()
        if model:
            return model.is_ready()
        return False
    
    def is_model_exported(self):
        # check if model is exported
        self._assert_project_is_ready()
        self._assert_model_is_ready()

        model = self._load_model()
        if model:
            return model.is_exported()
        return False
    
    def cleanup_model_dir(self):
        ''' Remove project directory '''
        # cleanup dataset dir
        model = self._load_model()
        if model:
            model.cleanup_model_dir()
    
    def cleanup_model_training(self):
        ''' Remove all files generated by TF '''
        model = self._load_model()
        if model:
            model.cleanup_train_eval()

    def cleanup_model_export(self):
        ''' Remove all file generated by export '''
        model = self._load_model()
        if model:
            model.cleanup_export()

    def cleanup_model(self):
        ''' Remove all model data'''
        self.cleanup_model_export()
        self.cleanup_model_training()
        self.cleanup_model_dir()
        model_path = Path(self.config.model_dir)
        if model_path.exists():
            try:
                model_path.rmdir()
            except:
                pass

    def train_and_eval(self):
        self._assert_project_is_ready()
        self._assert_dataset_is_ready()
        self._assert_model_is_ready()

        model = self._load_model()
        return model.train_and_eval()
    
    def continue_training(self):
        self._assert_project_is_ready()
        self._assert_dataset_is_ready()
        self._assert_model_is_ready()

        model = self._load_model()
        return model.continue_training()
    
    def model_checkpoints(self, name_only=True):
        '''
            Get list of model checkpoints available for export
        '''
        model = self._load_model()
        if model:
            return model.checkpoints(name_only=name_only)
        else:
            return []

    def export_model(self, checkpoint=None, force=False):
        '''
            Export model checkpoint for inference or
            as checkpoint for training of another model
        '''
        model = self._load_model()
        ckpt_names = model.checkpoints()
        if len(ckpt_names) < 1:
            raise ModelNotTrainedError('No model checkpoints available for export. Run Project.train_and_eval first.')

        return model.export(checkpoint=checkpoint, force=force)
    
    def get_n_iter(self):
        return self.config.n_iter

    def set_n_iter(self, n_iter):
        self.config.n_iter = n_iter
        if self.is_ready_model():
            model = self._load_model()
            model.n_iter = n_iter

        self.write_config()
        self.to_json()

    def set_batch_size(self, n_iter):
        self.config.batch_size = batch_size
        if self.is_ready_model():
            model = self._load_model()
            model.batch_size = batch_size

        self.write_config()
        self.to_json()
    # ==

    # ==
    # Inference
    # ==
    def detect(self, out_dir, image_path, output_types, padding=0, th=0.5):
        ''' Run detection and save outputs to files

            Parameters
            ----------
            out_dir : string
                path to output directory
            image_path: string
                path to single image or directory containing images
            output_type: list
                list of output types ['ibb', 'ebb', 'csv']
            padding: int
                padding to apply to bounding boxes in pixel
            th: float
                score threshold to still consider a box. Boxes with scores
        '''
        
        self._assert_model_is_exported()
        self._assert_dataset_is_ready()

        dataset = self._load_dataset()

        model = self._load_model()
        exported_model_path = model.get_exported_model_path()

        import ginjinn.core.tf_detector
        detector = ginjinn.core.tf_detector.TFDetector(
            exported_model_path,
            dataset.labelmap_path,
        )

        detector.run_detection(
            out_dir,
            image_path,
            output_types,
            padding=padding,
            th=th,
        )

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

        # save potentially updated config to json
        project.to_json()

        # update model config in case config.yaml was edited
        # since setup_model was called
        if project.is_ready_model():
            model = project._load_model()
            model.n_iter = project.config.n_iter
            model.batch_size = project.config.batch_size

        # print(project.config)

        return project

    def _assert_project_is_ready(self):
        if not self.is_ready():
            raise ProjectNotReadyError(
                'Project directory is not yet set up. Run Project.setup_project_dir first.'
            )
    
    def _assert_dataset_is_ready(self):
        if not self.is_ready_dataset():
            raise DatasetNotReadyError(
                'Dataset is not set up. Run Project.setup_dataset first.'
            )
    
    def _assert_model_is_ready(self):
        if not self.is_ready_model():
            raise ModelNotReadyError(
                'Model is not set up. Run Project.setup_model first.'
            )
    
    def _assert_model_is_exported(self):
        if not self.is_model_exported():
            raise ModelNotExportedError(
                'No exported model available. Run Project.export_model first.'
            )
    
    def _load_dataset(self):
        try:
            return TFDataset.from_directory(self.config.dataset_dir)
        except:
            return None
    
    def _load_model(self):
        try:
            return TFModel.from_directory(self.config.model_dir)
        except:
            return None

    @property
    def _project_files(self):
        return [
            self.config.config_path,
            self.config.project_json,
        ]

    def print_dataset_summary(self):
        try:
            self._assert_dataset_is_ready()
        except:
            print('Dataset is not set up.')
            return

        dataset = self._load_dataset()
        dataset.print_summary()