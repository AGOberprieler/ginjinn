from object_detection.utils import config_util
from object_detection.protos import train_pb2, preprocessor_pb2

AUGMENTATION_OPTIONS = {
    'flip_horizontal': {
        'active': bool,
    },
    'flip_vertical': {
        'active': bool,
    },
    'flip_90': {
        'active': bool,
    },
    'change_brightness': {
        'active': bool,
        'max_delta': float,
    },
    'change_contrast': {
        'active': bool,
        'min_delta': float,
        'max_delta': float,
    },
    'jitter_boxes': {
        'active': bool,
        'ratio': float,
    },
    
}

AO_TO_PPS = {
    'flip_horizontal': preprocessor_pb2.RandomHorizontalFlip,
    'flip_vertical': preprocessor_pb2.RandomVerticalFlip,
    'flip_90': preprocessor_pb2.RandomRotation90,
    'change_brightness': preprocessor_pb2.RandomAdjustBrightness,
    'change_contrast': preprocessor_pb2.RandomAdjustContrast,
    'jitter_boxes': preprocessor_pb2.RandomJitterBoxes,
}

AO_TO_PPOPTION = {
    'flip_horizontal': 'random_horizontal_flip',
    'flip_vertical': 'random_vertical_flip',
    'flip_90': 'random_rotation90',
    'change_brightness': 'random_adjust_brightness',
    'change_contrast': 'random_adjust_contrast',
    'jitter_boxes': 'random_jitter_boxes',
}

class InvalidAugmentationOptionError(Exception):
    pass

# TODO: this class is unnecessary, should refactor this
class TFAugmentation:
    ''' ginjinn tensorflow augmentation object

        Encanpsulates the construction of protobuf compatible
        objects from dictionaries.
    '''
    def __init__(self, augmentation_options):
        # print(augmentation_options)
        self.pps = preprocessing_steps_from_dict(augmentation_options)

def preprocessing_steps_from_dict(augmentation_options):
    pps = []
    for ao, cfg in augmentation_options.items():
        # print(ao, cfg)

        # check if valid augmentation option
        reference_cfg = AUGMENTATION_OPTIONS.get(ao, None)
        if not reference_cfg:
            msg = f'Unknown augmentation option {ao}'
            raise InvalidAugmentationOptionError(msg)
        
        # check if active
        active = cfg.get('active', None)
        if not active:
            continue

        # construct preprocessing step object
        pp = preprocessor_pb2.PreprocessingStep()
        pp_option = AO_TO_PPS[ao]()

        # check configs for ao
        for c_name, c_val in cfg.items():
            # skip active, since this is for ginjinn and not for tf
            if c_name == 'active':
                continue
            # ignore unknown configs
            if not reference_cfg.get(c_name):
                msg = f'Found unknown augmentation option: "{ao}: {c_name}" - options ignored.'
                print(msg)
                continue
            setattr(pp_option, c_name, c_val)

        # update preprocessing step object
        getattr(pp, AO_TO_PPOPTION[ao]).MergeFrom(pp_option)
        pps.append(pp)
    return pps