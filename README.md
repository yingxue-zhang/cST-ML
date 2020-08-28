# cST-ML
The codes and data of paper "cST-ML: Continuous Spatial-Temporal Meta-Learning for Traffic Dynamics Prediction".
The code and data are in the Dropbox folder: https://www.dropbox.com/sh/vimufus6jco46dv/AAA5UwYs57fNIuy55hMPb6IRa?dl=0.
The implementation is realized by Pytorch.


# Data Preparation
Since the whole dataset is huge, here we only provide sample datasets.

The sample data is located in Dropbox folder, please unzip data.zip and put the corresponding data files into cST-ML/code/.


# Requirements for Reproducibility
- Cuda 9.2
- pytorch 0.4.1
- Python 3.6.7
- Devices: NVIDIA GTX 1080 GPUs
- System: Ubuntu 16.04

# Training
Just take traffic speed estimation as an example:
1. `cd cST-ML/code/`.
2. Running the codes: `python train.py`. (The parameters are defined in train.py)
3. The parameters of the trained model will be saved in to cST-ML/code/Meta_VAE_train.

The training process for taxi inflow estimation is similar.
