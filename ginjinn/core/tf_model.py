import os
import sys
import json
import importlib_resources as resources
import subprocess

from pathlib import Path, PureWindowsPath

from ginjinn import config
from ginjinn.core import Configuration
from ginjinn.data_files import tf_config_templates, tf_script_templates

# TODO: maybe put those constants in another file
''' Model configuration files that are available out of the box'''
AVAILABLE_MODEL_CONFIGS = [
    f for f in resources.contents(tf_config_templates) if f.endswith('.config')
]

''' Models (names) that are available out of the box'''
AVAILABLE_MODELS = [
    # [:-7] removes '.config'
    f[:-7] for f in AVAILABLE_MODEL_CONFIGS
]

''' Mapping of model names to model configuration file names'''
MODEL_CONFIG_FILES = { m: f'{m}.config' for m in AVAILABLE_MODELS }

''' Mapping of model names (and configs) to urls where a coco-pretrained version can be downloaded '''
PRETRAINED_MODEL_URLS = {
    'faster_rcnn_resnet50_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
    'rfcn_resnet101_coco': 'http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz',
    'faster_rcnn_inception_resnet_v2_atrous_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz',
    'faster_rcnn_inception_v2_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',
    'faster_rcnn_resnet101_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz',
    'faster_rcnn_nas_coco': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',

    'ssd_mobilenet_v1_coco': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
    'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz',
    'ssd_mobilenet_v1_quantized_300x300_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz',
    'ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz',
    'ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz',
    'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
    'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync': 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
    'ssd_mobilenet_v2_coco': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
    'ssd_mobilenet_v2_quantized_300x300_coco': 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz',
    'ssd_mobilenet_v2_coco': 'http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz',
    'ssd_inception_v2_coco': 'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
}
PRETRAINED_MODEL_URLS.update({f'{m}.config':url for m, url in PRETRAINED_MODEL_URLS.items()})

# avoid circular imports
from ginjinn.utils.download_pretrained_model import download_and_extract_pretrained_model as dl_model

class PretrainedModelNotAvailableError(Exception):
    pass

class InvalidCheckpointException(Exception):
    pass

class PlatformNotSupportedError(Exception):
    pass

class TFModel:
    ''' TensorFlow model object

        Object for setting up a model directory, training the model, and exporting it.
    '''
    def __init__(self, model_dir):
        model_path = Path(model_dir).resolve()

        self.config = Configuration({
            'model_dir': str(model_path),
            'model_json': str(model_path.joinpath('model.json').resolve()),
            'model_config_path': str(model_path.joinpath('model.config').resolve()),
            'runscript_sh_path': str(model_path.joinpath('runscript.sh').resolve()),
            'runscript_cmd_path': str(model_path.joinpath('runscript.cmd').resolve()),
            'exportscript_sh_path': str(model_path.joinpath('exportscript.sh').resolve()),
            'exportscript_cmd_path': str(model_path.joinpath('exportscript.cmd').resolve()),
            'model_template': None,
            'record_train_path': None,
            'record_eval_path': None,
            'labelmap_path': None,
            'checkpoint_path': None,
            'use_checkpoint': True,
            'n_iter': 5000,
            'batch_size': 1,
            'augmentation': None,
        })
    
    def construct_model(
        self,
        model,
        record_train_path,
        record_eval_path,
        labelmap_path,
        checkpoint_path,
        use_checkpoint,
        augmentation_options,
        n_iter,
        batch_size,
    ):
        ''' Build model directory

            Builds the tensorflow model directory including configuration files and scripts
            for running model training and evalution.

            Parameters
            ----------
            model : string
                name of model that is provided or path to a tensorflow pipeline.config file
            record_train_path : string
                path to a tensorflow .record file, used for training
            record_eval_path : string
                path to a tensorflow .record file, used for evaluation
            labelmap_path : string
                path to the .pbtxt labelmap file corresponding to
                record_train_path and record_eval_path
            checkpoint_path : string
                path to a pretrained model checkpoint that is compatible with model
            use_checkpoint : boolean
                whether the pretrained model checkpoint should be used
            augmentation_options: dict
                dictionary containing augmentation options
            n_iter: int
                number of training steps
            batch_size: int
                number of images to evaluate at once
        '''

        from ginjinn.core.tf_model_configuration import TFModelConfigurationBuilder

        # check if input files exist
        # the check for checkpoint_path must be made later, since it might might a
        # downloadable checkpoint
        if not Path(record_train_path).exists():
            msg = f'Cannot find training data path "{record_train_path}".'
            raise Exception(msg)
        if not Path(record_eval_path).exists():
            msg = f'Cannot find evaluation data path "{record_eval_path}".'
            raise Exception(msg)
        if not Path(labelmap_path).exists():
            msg = f'Cannot find evaluation data path "{labelmap_path}".'
            raise Exception(msg)

        # check if model is name or path, and read template
        if model in AVAILABLE_MODEL_CONFIGS or model in AVAILABLE_MODELS:
            # model is available out of the box
            if model.endswith('.config'):
                _model = model[:-7] # remove '.config' from string
            else:
                _model = model
            model_template = resources.read_text(
                tf_config_templates,
                MODEL_CONFIG_FILES[_model],
            )
        else:
            model_path = Path(model).resolve()
            if not model_path.exists():
                msg = f'Cannot find model configuration at "{model}".'
                raise Exception(msg)
            # model is path to tensorflow pipeline configuration
            with model_path.open() as f:
                model_template = f.read()
        
        # check checkpoint_path and use_checkpoint. set flag if model should be downloaded later
        should_dl_ckpt = False
        if use_checkpoint:
            if not checkpoint_path:
                # check if model can be downloaded
                pretrained_model_url = PRETRAINED_MODEL_URLS.get(model, None)
                if not pretrained_model_url:
                    msg = f'No automatic download of checkpoint for model "{model}" available.'
                    raise PretrainedModelNotAvailableError(msg)
                else:
                    should_dl_ckpt = True
        
        # create model dir
        Path(self.config.model_dir).mkdir()

        # download model
        if should_dl_ckpt:
            # download model
            ckpt_dir = dl_model(pretrained_model_url, self.config.model_dir, rm=True)
            ckpt_prefix = str(Path(ckpt_dir).joinpath('model.ckpt').resolve())
            # set checkpoint_path to downloaded model
            checkpoint_path = ckpt_prefix
        
        # check if checkpoint is valid
        fs_exist = [
            Path(checkpoint_path + '.data-00000-of-00001').exists(),
            Path(checkpoint_path + '.index').exists(),
            Path(checkpoint_path + '.meta').exists(),
        ]
        if not all(fs_exist):
            msg = f'Invalid checkpoint. Make sure files with suffixes "data-00000-of-00001", "index", and "meta" exist for checkpoint "{checkpoint_path}"'
            raise InvalidCheckpointException(msg)

        # create model configuration file
        tfmc_builder = TFModelConfigurationBuilder(model_template)
        tfmc_builder.nclasses = _get_n_classes_from_labelmap(str(Path(labelmap_path).resolve()))
        tfmc_builder.record_train_path = str(Path(record_train_path).resolve())
        tfmc_builder.record_eval_path = str(Path(record_eval_path).resolve())
        tfmc_builder.labelmap_path = str(Path(labelmap_path).resolve())
        tfmc_builder.checkpoint_path = str(Path(checkpoint_path)) # no resolve, since path can be ''
        tfmc_builder.use_checkpoint = bool(use_checkpoint)
        # tfmc_builder.augmentation = augmentation_options # TODO: implement
        model_config = tfmc_builder.build_config_str()
        with Path(self.config.model_config_path).open('w') as f:
            f.write(model_config)

        # create runscripts
        self._construct_runscript_sh()
        self._construct_runscript_cmd()

        # update configuration
        self.config.model_template = model_template
        self.config.record_train_path = str(Path(record_train_path).resolve())
        self.config.record_eval_path = str(Path(record_eval_path).resolve())
        self.config.labelmap_path = str(Path(labelmap_path).resolve())
        self.config.use_checkpoint = bool(use_checkpoint)
        self.config.checkpoint_path = str(Path(checkpoint_path)) # no resolve, since path can be ''
        self.config.n_iter = int(n_iter)
        self.config.batch_size = int(batch_size)

        # write config to file
        self.to_json()
    
    def train_and_eval(self):
        # TODO: low priority: make output nicer
        if config.PLATFORM == 'Windows':
            runscript_path = self.config.runscript_cmd_path
        elif config.PLATFORM == 'Linux':
            runscript_path = self.config.runscript_sh_path
        else:
            msg = f'Platform "{config.PLATFORM}" is not supported.'
            raise PlatformNotSupportedError(msg)

        p = subprocess.Popen(
            runscript_path,
            cwd=config.RESEARCH_PATH
        )

        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()

    def to_json(self, fpath=None):
        '''
            Write configuration json file
        '''
        fpath = fpath or self.config.model_json
        with open(fpath, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_json(self, fpath=None):
        '''
            Load configuration from json file
        '''
        fpath = fpath or self.config.model_json
        with open(fpath) as f:
            self.config = Configuration(json.load(f))

    @classmethod
    def from_directory(cls):
        def from_directory(cls, model_dir):
        ''' Load TFModel object from existing directory
            The directory must have been successfully built
            using TFModel.construct_model
        '''

        # check if dataset_dir exists
        # TODO: print nicely formatted error
        Path(model_dir).resolve(strict=True)

        model = cls(model_dir)
        model.load_json()
        return model

    def _construct_runscript_sh(self, fpath=None):
        fpath = fpath or self.config.runscript_sh_path
        template = resources.read_text(tf_script_templates, 'runscript.sh')
        template = template.replace('<TF_RESEARCH_PATH>', str(Path(config.RESEARCH_PATH).as_posix()))
        template = template.replace('<TF_SLIM_PATH>', str(Path(config.SLIM_PATH).as_posix()))
        template = template.replace('<MODEL_CONFIG_PATH>', str(Path(self.config.model_config_path).as_posix()))
        template = template.replace('<MODEL_DIR>', str(Path(self.config.model_dir).as_posix()))
        template = template.replace('<N_ITER>', str(self.config.n_iter))
        template = template.replace('<BATCH_SIZE>', str(self.config.batch_size))
        template = template.replace('<LOG_TO_STDERR>', '--alsologtostderr')

        with Path(fpath).open('w') as f:
            f.write(template)
        
        os.chmod(fpath, 0o755)

        return fpath
    
    def _construct_runscript_cmd(self, fpath=None):
        fpath = fpath or self.config.runscript_cmd_path
        template = resources.read_text(tf_script_templates, 'runscript.cmd')
        template = template.replace('<TF_RESEARCH_PATH>', str(PureWindowsPath(config.RESEARCH_PATH)))
        template = template.replace('<TF_SLIM_PATH>', str(PureWindowsPath(config.SLIM_PATH)))
        template = template.replace('<MODEL_CONFIG_PATH>', str(PureWindowsPath(self.config.model_config_path)))
        template = template.replace('<MODEL_DIR>', str(PureWindowsPath(self.config.model_dir)))
        template = template.replace('<N_ITER>', str(self.config.n_iter))
        template = template.replace('<BATCH_SIZE>', str(self.config.batch_size))
        template = template.replace('<LOG_TO_STDERR>', '--alsologtostderr')

        with Path(fpath).open('w') as f:
            f.write(template)
        
        os.chmod(fpath, 0o755)

        return fpath


def _get_classdict_from_labelmap(file_path):
    from object_detection.utils import label_map_util
    return label_map_util.get_label_map_dict(file_path, use_display_name=True)

def _get_n_classes_from_labelmap(file_path):
    return len(_get_classdict_from_labelmap(file_path))