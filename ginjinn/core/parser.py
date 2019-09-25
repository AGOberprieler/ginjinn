import argparse
import sys

class Parser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)

parser = Parser(
    description='''
        Ginjinn is a pipeline for simplifying the setup, training, and deployment of
        object detection models for the extraction of structures
        from digitized herbarium specimens.
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

command_parsers = parser.add_subparsers(
    help='The following commands are available:',
    dest='command'
)
command_parsers.required = True

## new
new_parser = command_parsers.add_parser(
    'new',
    help='''
        Create a new project at the specified path.
    ''',
    description='''
        Create a new project at the specified path.
    ''',
)

new_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to the new project directory.',
)
new_parser.add_argument(
    '-f', '--force',
    help='''
        Force creation of new project. ATTENTION: Will remove the existing directory
        at project_dir!
    ''',
    action='store_true',
    default=False,
)

## auto
auto_parser = command_parsers.add_parser(
    'auto',
    help='''
        Run setup of dataset and model, model training, and export (of the most recent model checkpoint) based on the configuration in config.yaml.
    ''',
    description='''
        Run setup of dataset and model, model training and export (of the most recent model checkpoint) based on the configuration in config.yaml.
    ''',
)

auto_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory.',
)
auto_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite existing dataset, model, and model exports. ATTENTION: Will remove the existing data!
        This will also remove your trained model data!
    ''',
    action='store_true',
    default=False,
)

## setup_dataset
setup_dataset_parser = command_parsers.add_parser(
    'setup_dataset',
    help='''
        Run setup of the dataset based on the configuration in config.yaml.
    ''',
    description='''
        Run setup of the dataset based on the configuration in config.yaml.
    ''',
)

setup_dataset_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory.',
)
setup_dataset_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite existing dataset. ATTENTION: Will remove the existing data!
    ''',
    action='store_true',
    default=False,
)

## setup_model
setup_model_parser = command_parsers.add_parser(
    'setup_model',
    help='''
        Run setup of the model based on the configuration in config.yaml. Requires previous setup of the dataset.
    ''',
    description='''
        Run setup of the model based on the configuration in config.yaml. Requires previous setup of the dataset.
    ''',
)

setup_model_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires previous setup of the dataset.',
)
setup_model_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite existing model. ATTENTION: Will remove the existing data!
    ''',
    action='store_true',
    default=False,
)

## train model
train_parser = command_parsers.add_parser(
    'train',
    help='''
        Run model training and evaluation.
    ''',
    description='''
        Run model training and evaluation.
    ''',
)

train_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires previous setup of the dataset and model.',
)
train_parser.add_argument(
    '-c', '--continue_training',
    help='''
        Continue training of model. Use this if you interrupted training, increased the number of iterations,
        or changed the batch size.
    ''',
    action='store_true',
    default=False,
)
train_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite existing training data. ATTENTION: Will remove the existing data!
    ''',
    action='store_true',
    default=False,
)

## export model
export_parser = command_parsers.add_parser(
    'export',
    help='''
        Export trained model checkpoint.
    ''',
    description='''
        Export trained model checkpoint.
    ''',
)

export_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires previous training of the model.',
)
export_parser.add_argument(
    '-l', '--list_checkpoints',
    help='''
        List names of checkpoints available for export.
    ''',
    action='store_true',
    default=False,
)
export_parser.add_argument(
    '-c', '--checkpoint_name',
    help='''
        Name of the checkpoint used for export. By default the most recent checkpoint will be used.
    ''',
    type=str,
    default='',
)
export_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite existing export data. ATTENTION: Will remove the existing data!
    ''',
    action='store_true',
    default=False,
)

## detect
detect_parser = command_parsers.add_parser(
    'detect',
    help='''
        Run detection on images.
    ''',
    description='''
        Run detection on images.
    ''',
)

detect_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires previous export of the model.',
)
detect_parser.add_argument(
    'image_path',
    type=str,
    help='Path to directory containing images, or to a single image.',
)
detect_parser.add_argument(
    'out_dir',
    type=str,
    help='Path to output directory.',
)
detect_parser.add_argument(
    '-t', '--output_types',
    help='''
        The type of output to generate: separate images with bounding
        boxes for each class (-t ibb), extracted bounding boxes (-t ebb), ".csv" file with
        absolute and realtive bounding box locations and bounding box scores (-t csv).
        For multiple outputs, you have to repeat the command, e.g. -t ibb -t ebb for ibb and ebb.
    ''',
    nargs='+',
    choices=['ibb', 'ebb', 'csv'],
    action='append'
)
detect_parser.add_argument(
    '-p', '--padding',
    help='''
        Additional padding for the extraction of bounding boxes if output type "ebb" is selected.
    ''',
    type=int,
    default=0,
)
detect_parser.add_argument(
    '-s', '--score_threshold',
    help='''
        Threshold for the bounding box quality score. Bounding boxes below the threshold will not be
        present in the output. 
    ''',
    type=float,
    default=0.5,
)
detect_parser.add_argument(
    '-f', '--force',
    help='''
        Force overwrite output_dir. This will overwrite existing data, but NOT delete the directory.
    ''',
    action='store_true',
    default=False,
)

## status
status_parser = command_parsers.add_parser(
    'status',
    help='''
        Print project status.
    ''',
    description='''
        Print project status.
    ''',
)

status_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory.',
)

# list_models
list_models_parser = command_parsers.add_parser(
    'list_models',
    help='''
        Print available models.
    ''',
    description='''
        Print available models.
    ''',
)
list_models_parser.add_argument(
    '-d', '--downloadable',
    help='''
        List only automatically downloadable models.
    ''',
    action='store_true',
    default=False,
)

#== utils
utils_parser = command_parsers.add_parser(
    'utils',
    help='''
        Additional utility scripts.
    ''',
    description='''
        Additional utility scripts.
    ''',
)
utils_subparsers = utils_parser.add_subparsers(
    dest='utils_command',
    # required=True
)
utils_subparsers.required = True

# download_checkpoint
download_checkpoint_parser = utils_subparsers.add_parser(
    'download_checkpoint',
    help='''
        Download pretrained model checkpoint.
    ''',
    description='''
        Download pretrained model checkpoint.
    ''',
)

download_checkpoint_parser.add_argument(
    'model_name',
    type=str,
    help='''
        Name of the model for which the checkpoint should be downloaded.
        List available models with downloable checkpoints via "ginjinn list_models -d".
    ''',
)
download_checkpoint_parser.add_argument(
    'out_dir',
    type=str,
    help='''
        Directory, which the pretrained model checkpoint will be downloaded to.
    ''',
)

# data_summary
data_summary_parser = utils_subparsers.add_parser(
    'data_summary',
    help='''
        Print the summary of the dataset.
    ''',
    description='''
        Print the summary of the dataset.
    ''',
)

data_summary_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires the successfull setup of the dataset.',
)


# image_files
image_files_parser = utils_subparsers.add_parser(
    'image_files',
    help='''
        List the file paths of training or evaluation images files.
    ''',
    description='''
        List the file paths of training or evaluation images files.
    ''',
)

image_files_parser.add_argument(
    'project_dir',
    type=str,
    help='Path to existing project directory. Requires the successfull setup of the dataset.',
)

image_files_parser.add_argument(
    'type',
    type=str,
    help='''
        Type of images to list: training or evalution.
    ''',
    choices=['train', 'eval']
)

