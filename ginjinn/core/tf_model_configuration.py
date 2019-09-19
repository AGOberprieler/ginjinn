import tempfile
import tensorflow as tf

from pathlib import Path
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from ginjinn.core import Configuration

class RequiredConfigMissingError(Exception):
    pass

class ArchitectureNotSupportedError(Exception):
    pass

class TFModelConfigurationBuilder:
    ''' Tensorflow model configuration builder object

        Manages the population of tensorflow configuration templates.
    '''

    BASE_CONFIG = Configuration({
        'nclasses': None,
        'record_train_path': None,
        'record_eval_path': None,
        'labelmap_path': None,
        'checkpoint_path': None,
        'use_checkpoint': None,
        'augmentation': None,
    })

    def __init__(self, template):
        self.template = template
        self.reset()
    
    @property
    def nclasses(self):
        return self.config.nclasses
    
    @nclasses.setter
    def nclasses(self, nclasses):
        meta_architecture = self.model_config.WhichOneof("model")
        if meta_architecture == 'faster_rcnn':
            self.model_config.faster_rcnn.num_classes = nclasses
        elif meta_architecture == 'ssd':
            self.model_config.ssd.num_classes = nclasses
        else:
            msg = f'meta architecture "{meta_architecture}" not supported'
            raise ArchitectureNotSupportedError(msg)
            
        self.config.nclasses = nclasses

    @property
    def record_train_path(self):
        return self.config.record_train_path
    
    @record_train_path.setter
    def record_train_path(self, path):
        path = str(Path(path).resolve().as_posix())
        config_util.update_input_reader_config(
            self.pipeline_config,
            key_name='train_input_config',
            input_name=None,
            field_name='input_path',
            value=path,
        )
        self.config.record_train_path = path

    @property
    def record_eval_path(self):
        return self.config.record_eval_path
    
    @record_eval_path.setter
    def record_eval_path(self, path):
        path = str(Path(path).resolve().as_posix())
        config_util.update_input_reader_config(
            self.pipeline_config,
            key_name='eval_input_config',
            input_name=None,
            field_name='input_path',
            value=path,
        )
        self.config.record_eval_path = path

    @property
    def labelmap_path(self):
        return self.config.labelmap_path
    
    @labelmap_path.setter
    def labelmap_path(self, path):
        path = str(Path(path).resolve().as_posix())
        config_util._update_label_map_path(self.pipeline_config, path)
        self.config.labelmap_path = path

    @property
    def checkpoint_path(self):
        return self.config.checkpoint_path
    
    @checkpoint_path.setter
    def checkpoint_path(self, path):
        path = str(Path(path).resolve().as_posix())
        self.pipeline_config.get('train_config').fine_tune_checkpoint = path
        self.config.checkpoint_path = path

    @property
    def use_checkpoint(self):
        return self.config.use_checkpoint
    
    @use_checkpoint.setter
    def use_checkpoint(self, use_checkpoint):
        self.pipeline_config.get('train_config').from_detection_checkpoint = bool(use_checkpoint)
        self.config.use_checkpoint = bool(use_checkpoint)

    @property
    def augmentation(self):
        return self.config.augmentation
    
    @augmentation.setter
    def augmentation(self, augmentation_options):
        #TODO: implement
        pass

    def build_config_str(self):
        if self.config.nclasses is None:
            raise RequiredConfigMissingError('nclasses must be configured')
        if self.config.record_train_path is None:
            raise RequiredConfigMissingError('record_train_path must be configured')
        if self.config.record_eval_path is None:
            raise RequiredConfigMissingError('record_eval_path must be configured')
        if self.config.labelmap_path is None:
            raise RequiredConfigMissingError('labelmap_path must be configured')
        if self.config.checkpoint_path is None:
            raise RequiredConfigMissingError('checkpoint_path must be configured')
        if self.config.use_checkpoint is None:
            raise RequiredConfigMissingError('use_checkpoint must be configured')
        
        # TODO: implement augmentation options

        proto = config_util.create_pipeline_proto_from_configs(self.pipeline_config)
        return text_format.MessageToString(proto)

    def reset(self):
        self.config = self.BASE_CONFIG
        self.pipeline_config = get_configs_from_pipeline_string(self.template)
        self.model_config = self.pipeline_config.get('model')

def get_configs_from_pipeline_string(pipeline_config_string):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(pipeline_config_string, pipeline_config)

    return config_util.create_configs_from_pipeline_proto(pipeline_config)
