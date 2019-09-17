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