# GinJinn
GinJinn is an object detection command-line (Linux shell and Windows cmd) pipeline for the extraction of structures from digital images of herbarium specimens based on the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

GinJinn simplifies the management of object detection projects by automaticaly converting annotated images to TensorFlow record format and setting up model pipeline configurations, and provides an easy interface to training and export functionality. Additionally, GinJinn provides a command for instantly using trained and exported models for the detection of objects in new data.

See [Installation](#installation) for installation instructions on Windows and Linux. Mac is not supported.

See [Usage](#usage) for an introduction on how to use GinJinn, or look at the [example application](#example-application) for a practical introduction.

## Installation
Installation of the CPU version is easier, but the training and detection are (generally) also orders of magnitude slower. If possible, you should use the GPU version.
Do NOT install GPU and CPU version in the same Conda environment, otherwise you will get strange errors.

Installation instructions are available for:
 - [Linux](#linux): [CPU](#linux-cpu), [GPU](#linux-gpu)
 - [Windows 10](#windows): [CPU](#windows-cpu), [GPU](#windows-gpu)

### Linux
Both versions of GinJinn require [Conda](https://docs.conda.io/en/latest/), an open-source package management system for Python and R, which also includes facilities for environment management system. See the [official installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) for further information.

Run the following commands in your Linux terminal to install Conda:
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
) and execute it. You should check the option for appending Conda to PATH, since this will allow you to call the `conda` command from your terminal.

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
Make sure to activate your Conda environment via `conda activate MY_ENV_NAME` prior to running any ginjinn command.

### Setup, training, and export
The GinJinn pipeline consists of five steps:
1. Setup of the project directory:

	`ginjinn new my_project`

	This step generates the new directory "my_project". All subsequent steps will modify files and directories within "my_project". For now, there are two files: "config.yaml" and "project.json". "config.yaml" is the interesting one. Here, you can change data paths, model parameters, augmentation options, and so on. If you just want to try out GinJinn, you can just change the data paths ("image_dir", "annotation_path") and go to the next step.
2. Setup of the dataset:

	`ginjinn setup_dataset my_project`
	
	This step generates the input files to TensorFlow in the subdirectory "dataset", and prints a summary of you data so you can check, whether everything is working as expected.
	
3. Setup of the model:

	`ginjinn setup_model my_project`
	
	This step generates the model configuration files in the subdirectory "model". Here, the TensorFlow pipeline configuration file ("model/model.config") is generated and populated with the configurations you have made in the "config.yaml" files generated in step 1. You can modify the "model.config" file if you want fine control over the model parameters. Best have a look at the tutorials provided by the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), if you decide to do so.
	
4. Training and evaluation:

	`ginjinn train my_project`
	
	This steps runs simultaneous training and evaluation of the model. Checkpoints will be saved every 10 minutes and a maximum of five checkpoints will be saved before the oldest one will be removed. This is a limitation of the TensorFlow object detection API that will be removed in upcoming versions of GinJinn. While the training is running, you can monitor the progress and evalution metrics via the [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) tool. Just go to your "my_project/model" folder and run `tensorboard --logdir .` and open the address that is printed by tensorboard in your browser (generally "localhost:6006").
	
	If you should interrupt the training for some reason, you can restart it via `ginjinn train my_project -c`.
	
5. Model export:

	`ginjinn export my_project`
	
	Exports the latest model checkpoint for detection to "my_project/model/exported".

If you do not want to make any changes to any of the intermediary files, you can also run GinJinn with the `auto` command, which will run steps two through five without user interaction: `ginjinn auto my_project`

### Detection
After step five is successfully completed, the project can be used for detection of structures in images:

`ginjinn detect my_project image_dir out_dir -t ibb -t ebb -t csv`
	
This command runs object detection on all images in "image_dir" and writes the outputs to out_dir. The `-t` option describes the desired outputs, in this case: images with bounding boxes (`-t ibb`), extracted bounding boxes (`-t ebb`), and bounding box coordinates in a comma separated value file (`-t csv`).

For all GinJinn commands, there is a "help" option available, which will list possible options and their meaning. This help will be shown when calling the command with the `-h` or `--help` parameter.

## <a name="example-application">An example application: the *Leucanthemum* dataset</a>
We highly recommend using GinJinn GPU for this analysis; the CPU version will most likely be prohibitively slow. If you do not have a sufficient GPU and still want to reproduce this analysis you can use a computationally less demanding model (it will be noted in the following steps which on), which will only take around 24h to train, but will also lead to worse results.

##### Hardware used for the analysis:
- CPU: Intel Xeon CPU E5-2687W v3 @ 3.10GHz (10 cores)
- GPU: NVIDIA Quadro M4000
- Memory: 256 GB

##### Recommended hardware:

Originial analysis:
- CPU: faster is better
- Memory: at least 32 GB
- GPU: Any GPU with >= 8 GB memory

"Lite" version (CPU / GPU):
- CPU: faster is better / faster is better
- Memory: at least 24 GB if CPU only / 16 GB if GPU available
- GPU: None if CPU only / any GPU with at least 3 GB memory


##### The *Leucanthemum* dataset:
The dataset consists of 286 digitized herbarium specimens of either *Leucanthemum vulgare* or *L. ircutianum* in JPEG format and corresponding PASCAL VOC XML annotation. A single object class 'leaf', representing intact leaves (not overlapping or damaged) was annotated. In total, there are 886 leaf annotations in 243 of the the images. In 43 of the images no intact leaves could be found. Annotations were produced with [labelImg](https://github.com/tzutalin/labelImg).


##### Reproducing the analysis:
1. Download the dataset from [here](https://data.bgbm.org/dataset/gfbio/0033/gfbio0033.zip) and unzip it. It contains two folders: 'images', containing the JPEG images, and 'annotations', containing the PASCAL VOC XML annotations.

*Note*: The following commands assume that you have successfully installed GinJinn in a Conda environment, and require you to run the following commands in this environment in your terminal (shell, or cmd, depending on your OS)

2. Create a new GinJinn project:
	```
	ginjinn new leucanthemum_project
	```
	Generates the folder 'leucanthemum_project' in your working directory. 

3. Modify project configuration:
	Open 'leucanthemum_project/config.yaml' in a text-editor of your choice and replace 'ENTER PATH HERE' at `annotation_path` and `image_dir` with the *absolute* paths to the previously unzipped dataset folders. 
	To reproduce the *Leucanthemum* analysis, you also need to replace the `model` default option with 'faster_rcnn_inception_resnet_v2_atrous_coco, and setthe number of iterations to run `n_iter` to the value 12000. If you are using the CPU version, keep the default model and number of iterations.

4. Run the remaining pipeline steps:
		*ATTENTION: if you are not sure, whether your hardware is sufficient to handle model training, make sure to monitor the memory consumption of you system. TensorFlow will take as much memory as it needs, if that is more than your machine has available, your workstation will most likely freeze!*
	```
	ginjinn auto leucanthemum_project
	```
	This command will sequentially run dataset setup (`ginjinn setup_dataset`), model setup (`ginjinn setup_model`), model training (`ginjinn train`), and model export (`ginjinn export`).
	
	You can and should monitor progress of the model training via [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard). Simply run `tensorboard --logdir leucanthemum_project` in a new shell in the same conda environment. This will host a tiny webserver on your local machine that you visit in your browser at the link printed to the console by TensorBoard. On default that is 'localhost:6006'. 

6. You can now use the GinJinn project for detection of leaves:
	```
	ginjinn detect leucanthemum_project PATH_TO_IMAGES OUT_DIR -t csv -t ibb -t evv
	```
	Here, PATH_TO_IMAGES is the path to a directory containing images (JPEG or PNG) - for this trial just select the *Leucanthemum* images folder - and OUT_DIR is the path of a directory that will be newly generated, where the outputs will be saved. The `-t` option tells GinJinn the kinds of outputs you want to get. `-t ibb` means images with bounding boxes, `-t ebb` means extracted bounding boxes, and `-t csv` means bounding boxes in CSV format. 
