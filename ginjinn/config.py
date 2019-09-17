import sys
import os
import platform
import pathlib import Path

CONDA_PREFIX = os.environ.get('CONDA_PREFIX')

MODELS_PATH = str(Path(CONDA_PREFIX).joinpath('models').resolve())
RESEARCH_PATH = str(Path(MODELS_PATH).joinpath('research').resolve())
SLIM_PATH = str(Path(RESEARCH_PATH).joinpath('slim').resolve())

PLATFORM = platform.system()
