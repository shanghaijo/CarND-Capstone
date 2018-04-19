# Model Training for the Traffic Light Classifier

## Installation

### *NIX Instructions

Follow the instructions found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

### Windows Instructions

1. Open a priveledged command prompt in this directory
1. Clone the `tensorflow/models` repo:
    ```cmd
    git clone https://github.com/tensorflow/models.git
    ```
1. Install [python3](https://www.python.org/)
1. Install tensorflow:
    ```cmd
    pip install tensorflow
    ```
    or
    ```cmd
    pip install tensorflow-gpu
1. Install python libraries:
    ```cmd
    pip install Cython
    pip install pillow
    pip install lxml
    pip install jupyter
    pip install matplotlib
    ```
1. Install the [Windows-compatible fork of the pycocotools API](https://github.com/philferriere/cocoapi) according to it's readme (requires Visual Studio 2015 or higher with C++ packages installed):
    ```cmd
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    ```
1. Download [protoc](https://github.com/google/protobuf/releases/), place it in a directory of your choice, and add its location to your `PATH` environment variable
1. Run Protobuf compilation:
    ```cmd
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    ```
1. Test the installation by running:
    ```cmd
    set PYTHONPATH=%PYTHONPATH%;%cd%;%cd%/slim
    python object_detection/builders/model_builder_test.py
    ```
