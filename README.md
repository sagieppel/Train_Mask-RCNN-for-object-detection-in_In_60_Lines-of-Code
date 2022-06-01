# Train_Mask-RCNN-for-object-detection-in_In_60_Lines-of-Code

Toturial for this code can be seen here:
https://medium.com/@sagieppel/train-mask-rcnn-net-for-object-detection-in-60-lines-of-code-9b6bbff292c3

# Dataset
This code is for using training mask RCNN with pytorch and the [LabPics Version 2 datast](https://zenodo.org/record/4736111#.YpfnEqjMK3A).

# Requirement 
This demand pytorch 1.1 and opencv. Runnig it with newere pytorch version might cause some isseus.
Pytorch installation instructions are available at:

https://pytorch.org/get-started/locally/

OpenCV can be installed using:

pip install opencv-python
# Requirements


## Setting enviroment with conda
1) Install [Anaconda](https://www.anaconda.com/download/)
2) Create a virtual environment with the required dependencies ([Pytorch](https://pytorch.org/), torchvision and OpenCV): *conda env create -f environment.yml*
3) Activate the virtual environment: *conda activate  maskRcnn*

## Hardware
For using the trained net, no specific hardware is needed, but the net will run much faster on GPU.

For effectively training the net a good GPU is needed.

# Train model

Code with pre trained model weights can be download from here: https://icedrive.net/s/v6xR9fbh6F24zXSW8bBt6aji6F6x

