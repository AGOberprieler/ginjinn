# GinJinn
GinJinn is an object detection command-line (Linux shell and Windows cmd) pipeline for the extraction of structures from digital images of herbarium specimens based on the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

GinJinn simplifies the management of object detection projects by automaticcaly converting annotated images to tensorflow record format and setting up model pipeline configurations, and provides an easy interface to training and export functionality. Additionally, GinJinn provides a command to instantly use trained and exported models for the detection of objects in new data.

See [Installation](#installation) for installation instructions on Windows and Linux. MAC is not supported.

See [Usage](#usage) for an introduction on how to use GinJinn, or look at the [example application](#example-application) for a pratical introduction.

## Installation
GinJinn supports the standard TensorFlow CPU version, as well as TensorFlow-GPU.
Installation of the CPU version is easier, but the training and detection are (compared to the GPU version with a recent GPU) orders of magnitude slower. If possible, you should use the GPU version.

Do NOT install GPU and CPU version in the same conda environment, otherwise you will get strange errors.

All installation commands must be run from your systems command line. For Linux that is the shell/bash, and for windows that is cmd.

### CPU version
1. Install Conda. [Conda](https://docs.conda.io/en/latest/) is an open-source package management system for Python and R, which also includes an environment management system.
	- Windows ([official guide](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)):
		1. Download miniconda Python 3.7 GUI installer from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
		2. Run the GUI installer via double click. Select append to Path when asked.
	- Linux ([official guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)):
		1. Download miniconda Python 3.7 via:
			```bash
			wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
			```
		2. Install miniconda. Select append to .bashrc/Path if asked:
			```bash
			bash Miniconda3-latest-Linux-x86_64.sh
			```
2. Setup conda environment
	```
	conda create -n ginjinn-cpu python=3.6
	```
3. Activate conda environment. You will have to run this command every time you want to use ginjinn after closing your terminal. The remaining installation assumes that this environment is activated.
	```
	conda activate ginjinn-cpu
	```
4. Install GinJinn
	```
	conda install ginjinn -c conda-forge -c AGOberprieler 
	```
5. Install tensorflow
Attention: never install CPU and GPU version in the same conda environment!
	```
	pip install tensorflow
	```
6. (Windows only) Install pycocotools:
On linux, pycocotools are automatically installed with ginjinn. On Windows, you will have to install them manually via pip. Make sure, you have the "Visual C++ 2015 build tools" ([download here](https://go.microsoft.com/fwlink/?LinkId=691126)) installed before executing the following command.
	```
	pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
	```

### GPU version
1. Install Conda. [Conda](https://docs.conda.io/en/latest/) is an open-source package management system for Python and R, which also includes an environment management system.
	- Windows ([official guide](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)):
		1. Download miniconda Python 3.7 installer from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
		2. Run installer. Select the append to Path option.
	- Linux ([official guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)):
		1. Download miniconda Python 3.7 via:
			```bash
			wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
			```
		2. Install miniconda. Select append to .bashrc if asked:
			```bash
			bash Miniconda3-latest-Linux-x86_64.sh
			```
2. Install software requirements listed at [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for tensorflow 1. This will include proprietary NVIDIA drivers and libraries and requires you to create an account at NVIDIA. Make sure your GPU is supported!

3. Setup conda environment.
	```
	conda create -n ginjinn-gpu python=3.6
	```

4. Activate conda environment
	You will have to run this command every time you want to use ginjinn after closing your terminal. The remaining installation assumes that this environment is activated.
	```
	conda activate ginjinn-gpu
	```
5. Install GinJinn
	```
	conda install ginjinn -c conda-forge -c AGOberprieler 
	```
6. Install tensorflow-gpu
Attention: never install CPU and GPU version in the same conda environment!
	```
	pip install tensorflow-gpu
	```
7.  (Windows only) Install pycocotools:
On linux, pycocotools are automatically installed with ginjinn. On Windows, you will have to install them manually via pip. Make sure, you have the "Visual C++ 2015 build tools" ([download here](https://go.microsoft.com/fwlink/?LinkId=691126)) installed before executing the following command.
	```
	pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
	```

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

If you don't want to make any changes to any of the intermediary files, you can also run GinJinn with the `auto` command, which will run steps two to five without user interaction:

	`ginjinn auto my_project`

### Detection
After step five successfully completed, the project can be used for detection of structures in images:

`ginjinn detect my_project image_dir out_dir -t ibb -t ebb -t csv`
	
This command runs object detection on all images in "image_dir" and writes the outputs to out_dir. The `-t` option describes the desired outputs, in this case: images with bounding boxes (`-t ibb`), extraced bounding boxes (`-t ebb`), and bounding box coordinates in a comman separated value file (`-t csv`).

For all GinJinn commands, there is a "help" option available, which will list possible options and their meaning. This help will be shown when calling the command with the `-h` or `--help` parameter.

## <a name="example-application">An example application: the *Leucanthemum* dataset</a>
### Not yet published
