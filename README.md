# GinJinn
GinJinn is an object detection command-line (Linux shell and Windows cmd) pipeline for the extraction of structures from digital images of herbarium specimens based on the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

GinJinn simplifies the management of object detection projects by automaticcaly converting annotated images to tensorflow record format and setting up model pipeline configurations, and provides an easy interface to training and export functionality. Additionally, GinJinn provides a command to instantly use trained and exported models for the detection of objects in new data.

See [Installation](#installation) for installation instructions on Windows and Linux. MAC is not supported.

See [Usage](#usage) for an introduction on how to use GinJinn, or look at the [example application](#example-application) for a pratical introduction.

## Installation
Installation of the CPU version is easier, but the training and detection are (generally) also orders of magnitude slower. If possible, you should use the GPU version.
Do NOT install GPU and CPU version in the same conda environment, otherwise you will get strange errors.

Installation instructions are available for:
 - [Linux](#linux): [CPU](#linux-cpu), [GPU](#linux-gpu)
 - [Windows 10](#windows): [CPU](#windows-cpu), [GPU](#windows-gpu)

### Linux
Both versions of GinJinn require [Conda](https://docs.conda.io/en/latest/), an open-source package management system for Python and R, which also includes facilities for environment management system. See the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) for further information.

Run the following commands in your linux terminal to install Conda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
This will download the installer via `wget` and run the installation. When asked if you want to initialize conda for you terminal, answer with 'yes'. You will need to restart your terminal after the installation of Conda.

#### Linux CPU
Execute the following commands from your Linux shell.
1. Create an [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):
	```bash
	conda create -n ginjinn-cpu python=3.6 tensorflow=1.14.0
	``` 
	This command will setup a new Conda environment named 'ginjinn-cpu' with the packages Python 3.6 and TensorFlow 1.14.

2. Activate the Conda environment:
	```bash
	conda activate ginjinn-cpu
	```
	This command activates the Conda environment we previously created. This means that packages that are installed via `conda install` will be installed to this environment and do not influence the remaining system.
	**IMPORTANT**: All subsequent installation steps require that the ginjinn-cpu conda environment is active.

3. Install GinJinn for CPU:
	```bash
	conda install -c conda-forge -c AGOberprieler ginjinn
	```
	This command installs GinJinn and its dependencies into the active environment. The `-c` parameters specify additional channels/repositories that should be used.
	
	GinJinn is now ready to use.

#### Linux GPU
Execute the following commands from your Linux shell.
1. Install all requirements listed at [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for TensorFlow version 1.14. Make also sure your GPU is supported.

2. Create an [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):
	```bash
	conda create -n ginjinn-gpu python=3.6 tensorflow-gpu=1.14.0
	``` 
	This command will setup a new Conda environment named 'ginjinn-cpu' with the packages Python 3.6 and TensorFlow GPU 1.14.

3. Activate the Conda environment:
	```bash
	conda activate ginjinn-gpu
	```
	This command activates the Conda environment we previously created. This means that packages that are installed via `conda install` will be installed to this environment and do not influence the remaining system.
	**IMPORTANT**: All subsequent installation steps require that the ginjinn-gpu conda environment is active.

4. Install GinJinn for GPU:
	```bash
	conda install -c conda-forge -c AGOberprieler ginjinn-gpu
	```
	This command installs GinJinn and its dependencies into the active environment. The `-c` parameters specify additional channels/repositories that should be used.
	
	GinJinn is now ready to use.

### Windows
Both versions of GinJinn require [Conda](https://docs.conda.io/en/latest/), an open-source package management system for Python and R, which also includes facilities for environment management system. See the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html) for further information.

To install Conda, download the installer from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
) and execute it. You should check the option for appending conda to PATH, since this will allow you to call the `conda` command from your terminal.

#### Windows CPU
All following commands should be executed in a standard CMD terminal.

1. Create an [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):
	```batch
	conda create -n ginjinn-cpu python=3.6 tensorflow=1.14.0
	``` 
	This command will setup a new Conda environment named 'ginjinn-cpu' with the packages Python 3.6 and TensorFlow 1.14.

2. Activate the Conda environment:
	```batch
	conda activate ginjinn-cpu
	```
	This command activates the Conda environment we previously created. This means that packages that are installed via `conda install` will be installed to this environment and do not influence the remaining system.
	**IMPORTANT**: All subsequent installation steps require that the ginjinn-cpu conda environment is active.

3. Install GinJinn for CPU:
	```batch
	conda install -c conda-forge -c AGOberprieler ginjinn
	```
	This command installs GinJinn and its dependencies into the active environment. The `-c` parameters specify additional channels/repositories that should be used.

4. Install [pycocotools](https://github.com/philferriere/cocoapi) for windows:
	```batch
	pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
	```
	This package is necessary for the COCO evaluation metrics that the TensorFlow object detection API uses.
	
	GinJinn is now ready to use.

#### Windows GPU
1. Install all requirements listed at [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for TensorFlow version 1.14. Make also sure your GPU is supported.
2. Create an [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html):
	```batch
	conda create -n ginjinn-gpu python=3.6 tensorflow-gpu=1.14.0
	``` 
	This command will setup a new Conda environment named 'ginjinn-cpu' with the packages Python 3.6 and TensorFlow GPU 1.14.

3. Activate the Conda environment:
	```batch
	conda activate ginjinn-gpu
	```
	This command activates the Conda environment we previously created. This means that packages that are installed via `conda install` will be installed to this environment and do not influence the remaining system.
	**IMPORTANT**: All subsequent installation steps require that the ginjinn-gpu conda environment is active.

4. Install GinJinn for GPU:
	```batch
	conda install -c conda-forge -c AGOberprieler ginjinn-gpu
	```
	This command installs GinJinn and its dependencies into the active environment. The `-c` parameters specify additional channels/repositories that should be used.

5. Install [pycocotools](https://github.com/philferriere/cocoapi) for windows:
	```batch
	pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
	```
	This package is necessary for the COCO evaluation metrics that the TensorFlow object detection API uses.
	
	GinJinn is now ready to use.

## Usage
Make sure to activate your conda environment via `conda activate MY_ENV_NAME` prior to running any ginjinn command.

### Setup, Training, and Export
The GinJinn pipeline consists of five steps:
1. Setup of the project directory:

	`ginjinn new my_project`

	This step generates the new directory "my_project". All subsequent steps will modify files and directories within "my_project". For now, there are two files: "config.yaml" and "project.json". "config.yaml" is the interesting one. Here, you can change data paths, model parameters, augmentation options, and so on. If you just want to try out GinJinn, you can just change the data paths ("image_dir", "annotation_path") and go to the next step.
2. Setup of the dataset:

	`ginjinn setup_dataset my_project`
	
	This step generates the input files to TensorFlow in the subdirectory "dataset", and prints a summary of you data so you can check, whether everything is working as expected.
	
3. Setup of the model:

	`ginjinn setup_model my_project`
	
	This step generates the model configuration files in the subdirectory "model". Here, the TensorFlow pipeline configuration file ("model/model.config") is generated and populated with the configurations you made in the "config.yaml" files generated by step 1. You can modify the "model.config" file if you want fine control over the model parameters. Best have a look at the tutorials provided by the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), if you decide to do so.
	
4. Training and evalution:

	`ginjinn train my_project`
	
	This steps runs simultaneous training and evaluation of the model. Checkpoints will be saved every 10 minutes and a maximum of 5 checkpoints will be saved before the oldest one will be removed. This is limitation, that will be removed in upcoming versions of GinJinn. While the training is running you can monitor the progress and evalution metrics via the [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) tool. Just go to your "my_project/model" folder and run `tensorboard --logdir .` and open the address that is printed by tensorboard in your browser (generally "localhost:6006").
	
	If you should interrup the training for some reason, you can restart it via `ginjinn train my_project -c`.
	
5. Model export:

	`ginjinn export my_project`
	
	Exports the latest model checkpoint for detection to "my_project/model/exported".

If you don't want to make any changes to any of the intermediary files, you can also run GinJinn with the `auto` command, which will run steps two to five without user interaction: `ginjinn auto my_project`

### Detection
After step five successfully completed, the project can be used for detection of structures in images:

`ginjinn detect my_project image_dir out_dir -t ibb -t ebb -t csv`
	
This command runs object detection on all images in "image_dir" and writes the outputs to out_dir. The `-t` option describes the desired outputs, in this case: images with bounding boxes (`-t ibb`), extraced bounding boxes (`-t ebb`), and bounding box coordinates in a comman separated value file (`-t csv`).

For all GinJinn commands, there is a "help" option available, which will list possible options and their meaning. This help will be shown when calling the command with the `-h` or `--help` parameter.

## <a name="example-application">An example application: the *Leucanthemum* dataset</a>
### Not yet published
