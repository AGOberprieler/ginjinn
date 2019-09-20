#!/usr/bin/env python

import sys
from pathlib import Path

from ginjinn import config
from ginjinn.core import parser
from ginjinn.core import Project

def main():
    # print(config.PLATFORM)
    # print(config.MODELS_PATH, config.RESEARCH_PATH, config.SLIM_PATH)

    args = parser.parse_args()
    # print(args)

    # new - create new project
    if args.command == 'new':
        p = Project(args.project_dir)
        try:
            p.setup_project_dir(force=args.force)
            print(
                f'Created new project at \"{Path(args.project_dir).resolve()}\".\n' +
                f'Configurate the project via \"{Path(args.project_dir).joinpath("config.yaml").resolve()}\".'
            )
        except FileExistsError:
            msg = 'Directory at "{}" already exists. Select another directory or rerun with --force.'.format(
                args.project_dir
            )
            raise FileExistsError(msg)

    # auto - run setup, training and export
    elif args.command == 'auto':
        p = Project.from_directory(args.project_dir)
        if args.force:
            p.cleanup_data_model_export()

        p.setup_dataset()
        p.print_dataset_summary()
        print('Successfully setup dataset ...')
        p.setup_model()
        print('Successfully setup model ...')
        p.train_and_eval()
        print('Successfully trained model ...')
        p.export_model()
        print('Successfully exported model ...')
        print('Project ready for detection.')

    # setup_dataset
    elif args.command == 'setup_dataset':
        p = Project.from_directory(args.project_dir)
        if args.force:
            p.cleanup_dataset_dir()

        p.setup_dataset()
        print('Successfully setup dataset.')
        p.print_dataset_summary()

    # setup_model
    elif args.command == 'setup_model':
        p = Project.from_directory(args.project_dir)
        if args.force:
            p.cleanup_model()
        
        p.setup_model()
        print('Successfully setup model.')

    # train
    elif args.command == 'train':
        p = Project.from_directory(args.project_dir)
        if args.force:
            p.cleanup_model_training()
        
        if args.continue_training:
            p.continue_training()
        else:
            try:
                p.train_and_eval()
            except:
                raise Exception('Model was already trained. Run command with -f/--force to overwrite previous training, or with -c/--continue_training to continue the previous training.')

        print('Successfully trained model.')

    # export
    elif args.command == 'export':
        p = Project.from_directory(args.project_dir)

        if args.list_checkpoints:
            ckpts = p.model_checkpoints()
            msg = 'Available model checkpoints are:\n'
            msg += '\n'.join([f'\t{c}' for c in ckpts])
            print(msg)
            return

        if args.force:
            p.cleanup_model_export()
        
        ckpt = args.checkpoint_name or None
        try:
            p.export_model(ckpt)
        except:
            raise Exception('Model was already exported. Run command with -f/--force to overwrite previous exports.')

        print('Successfully exported model.')

    # detect
    elif args.command == 'detect':
        # argparse outputs this as list of lists, so we have to convert it to a list
        if not args.output_types:
            output_types = ['ibb']
        else:
            output_types = [t[0] for t in args.output_types]
        
        p = Project.from_directory(args.project_dir)

        out_path = Path(args.out_dir)
        if out_path.exists():
            if not args.force:
                raise Exception('Output directory already exists. Select another directory or rerun with -f/--force to overwrite the data.')
        # def detect(self, out_dir, image_path, output_types, padding=0, th=0.5):
        p.detect(
            args.out_dir,
            args.image_path,
            output_types,
            padding = args.padding,
            th = args.score_threshold
        )
        print('Successfully ran detection.')
        print(f'Detection data written to {args.out_dir}')

    # status
    elif args.command == 'status':
        p = Project.from_directory(args.project_dir)
        msg = f'Status for project "{p.config.project_dir}":\n'
        msg += f'\tproject setup:\t\t\t\t{p.is_ready()}\n'
        msg += f'\tdataset setup:\t\t\t\t{p.is_ready_dataset()}\n'
        msg += f'\tmodel setup:\t\t\t\t{p.is_ready_model()}\n'
        msg += f'\tmodel (at least partially) trained:\t{bool(p.model_checkpoints())}\n'
        msg += f'\tmodel exported:\t\t\t\t{p.is_model_exported()}\n'
        if p.is_model_exported():
            msg += '\nProject is ready for detection.'
        print(msg)

    # list_models
    elif args.command == 'list_models':
        from ginjinn.core.tf_model import AVAILABLE_MODELS, AVAILABLE_MODELS_DOWNLOADABLE
        
        if args.downloadable:
            msg = 'Available models with downloadable checkpoint:\n'
            msg += '\n'.join([f'\t- {name}' for name in AVAILABLE_MODELS_DOWNLOADABLE])
        else:
            msg = 'Available models:\n'
            msg += '\n'.join([f'\t- {name}' for name in AVAILABLE_MODELS])
        print(msg)

if __name__ == '__main__':
    main()