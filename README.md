# GinJinn
GinJinn is an object detection pipeline for the extraction of structures from digital images of herbarium specimens based on the [TensorFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

GinJinn simplifies the management of object detection projects by automaticcaly converting annotated images to tensorflow record format and setting up model pipeline configurations, and provides an easy interface to training and export functionality. Additionally, GinJinn provides a command to instantly use trained and exported models for the detection of objects in new data.

See [Installation](#installation) for installation instructions on Windows and Linux. MAC is not supported.
See [Usage](#usage) for an example application of GinJinn

## Installation
GinJinn supports the standard TensorFlow CPU version, as well as TensorFlow-GPU.
Installation of the CPU version is easier, but the training and detection are (compared to the GPU version with a recent GPU) orders of magnitude slower. If possible, you should use the GPU version.
Do NOT install GPU and CPU version in the same conda environment, otherwise you will get strange errors.

### CPU version
1. Install conda:
	- Windows:
		1. Download miniconda Python 3.7 installer from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
		2. Run installer. Select append to Path when asked.
	- Linux:
		1. Download miniconda Python 3.7 via:
			`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
		2. Install miniconda. Select append to .bashrc/Path if asked:
			`bash Miniconda3-latest-Linux-x86_64.sh`
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
1. Install conda:
	- Windows:
		1. Download miniconda Python 3.7 installer from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
		2. Run installer. Select the append to Path option.
	- Linux:
		1. Download miniconda Python 3.7 via:
			`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
		2. Install miniconda. Select append to .bashrc if asked:
			`bash Miniconda3-latest-Linux-x86_64.sh`
2. Install software requirements listed at [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for tensorflow 1. This will include proprietary NVIDIA drivers and libraries and requires you to create an account at NVIDIA. Make sure your GPU is supported!

3. Setup conda environment
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
