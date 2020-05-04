# BigDataForHealth

### General setup
1. Use Conda to set up a virtual environment and install all the required packages, including Torch, TorchVision, OpenCV, pandas, numpy, matplotlib, PySpark, PIL, sklearn, etc
2. Use Ubuntu. Windows or Mac could cause problems
3. Preferably use a GPU that supports CUDA and has more than 8GB of memory (this affects the max batch_size)

### To generate preprocessed images
1. Make sure that the `CheXpert-v1.0-small` folder is in the root directory of the repository
2. Run `python preprocess.py`  to generate all cropped frontal images. The cropped image is named as `temp` + the original file name and is stored at the same directory as the original image

### To train the model
1. Replace the `model` variable with the model class you want to train, default is Densenet
1. Run `python train.py` to train the model to get the BCE loss graph

### To test the model
1. Replace the model path with the model paths you want to compare. By default, it compares the Densenet model and the Resnet model
1. Run `python test.py`

### To generate a heatmap
1. Replace `input_image` with the path to the image that you want to generate a heatmap for
2. Replace path_to_model with the path to the model you want to use. So far it only works for DenseNet

