# Hardware-optimized machine learning model comparison for human activity recognition using the HARTH dataset

Code for models used in this capstone paper under the University of the Philippines Microelectronics and Microprocessors Laboratory for BS Electronics Engineering.

## Features

- Preprocessing and feature extraction for HARTH and HAR70 data (CSV format)
- Configurable SVM (libsvm) and CNN (Tensorflow/Keras) implementations in Python for inference and training
- hls4ml (https://github.com/fastmachinelearning/hls4ml) integration for CNN synthesis in Vivado HLx
- C implementation of SVM inference for Vivado HL

## Prerequisites

- Tensorflow 2.11 (CUDA 11.3. CUDNN 8.1)
- Vivado HLx 2019.1 
- NumPy, SciPy, Pandas for preprocessing
- Seaborn for statistics

## Usage

1. Clone this repository along with submodules, and install prerequisites. Make sure that Vivado is enabled if you will be doing synthesis.

```sh
git clone --recursive https://github.com/HAR-ML-Embedded-Microlab/classifiers
cd classifiers
source install_requirements.sh
source $VIVADO_DIR/settings64.sh 
```

2. For preprocessing and feature extraction (artifacts stored in work folder by default)
```sh
source env/bin/activate
python data/preprocess_data.py --window-size 50
python data/feature_extraction.py --window-size 50
```

3. Use the corresponding scripts and options for the desired model
```sh
python cnn/cnn_model.py --name cnn_model --window-size 50
python cnn/cnn_train.py --name cnn_model 
python cnn/cnn_crossval.py --name cnn_model
python cnn/cnn_synth.py --name cnn_model
```
