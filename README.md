# Comparison of traffic sign classification performance using images of different color spaces

## About
Recent advances in deep learning have brought major breakthroughs in the field of machine learning. Especially the field of image classification has been benefiting from the development of convolutional neural networks. Image classification is at the core of many problems in robotics and autonomous driving and extensive research has been addressed to further identify high performing convolutional network architectures and improve their training pipelines. Very little on the other hand has been done to investigate the influences of image preprocessing, such as color space transformations. To address this issue a CIFAR10-ConvNet for image classification is trained on different color space transformed versions of an image dataset of German traffic signs. Precision, recall and f-1 score are then evaluated to provide researchers with insights on the influences of color space transformations on image classification performance to consider for their future research.

## Dependencies
* tensorflow 2
* numpy
* matplotlib
* pandas
* google colab
* scikit-learn

## Overview

Following steps were taken during this project and will be further elaborated on:
1. Step Environment Setup and Load Data
2. Step Data Pipeline and Preprocessing
3. Step Model Setup
4. Step Model Training 
5. Step Model Evaluation

### 1. Step - Environment Setup and Load Data
All steps are performed in google colab (https://colab.research.google.com/) in order to achieve faster training using its free GPU-support.
The german traffic sign detection benchmark dataset is used for this project (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and was downloaded from (https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

The dataset consist of 50.000 images of german traffic signs belonging to 43 different classes.
All images are provided in .ppm format.

Example:

![example](http://benchmark.ini.rub.de/Images/00005_00000.jpg)
![example2](http://benchmark.ini.rub.de/Images/00000_00023.jpg)
![example2](http://benchmark.ini.rub.de/Images/00001_00017.jpg)

### 2. Step - Data Pipeline and Preprocessing
A train/test split doesn't have to be performed, since a predefined split is provided. The train set is further split into train and validation set.
For each color space and train/validation/test set a seperate ImageDataGenerator is created and following preprocessing steps defined:
* Transform to colorspace (rgb, grayscale, hsv, yuv, yiq)
* Resize to 64x64 pixels
* Scale pixel values from 0 to 1

### 3. Step - Model Setup
A simple CIFAR10-Convnet architecture is defined using the tensorflow keras api. 
The network consists of two relu activated convolutional layers followed by a max-pooling layer and a dropout layer to increase robustness followed by the same setup of convolutional, max pooling and dropout layers.
Dense layers are make up the tail and conclide the models architecture.

Network Architecture:
![network architecture](https://github.com/philipoedi/Color-Spaces-and-Conv-Net/blob/master/plots/model_structure.PNG?raw=true)

### 4. Step - Model Training
A new instance of the above described model structure is created for each of the observed color spaces and trained for 20 epochs, using the Adam Optimizer with a learning rate of 0.001 to minimize the categorical crossentropy.

The per epoch achieved training- and validation loss values, as well as the models accuracy are logged to seperate files saved on my personal google-drive.

![training-history](https://github.com/philipoedi/Color-Spaces-and-Conv-Net/blob/master/plots/training_history.png?raw=true)

### 5. Step - Model Evaluation
The test set is used for model evaluation. Each model is used to predict class labels. The predicted class labels are then join to the ground truth metadata table and also saved to the results.csv on my gdrive. 
Ground truth and predicted classes are then compared using scikit-learn and their corresponding precision, recall and f1-scores calculated. First a score for the individual traffic sign class is calculated, which are then aggregated to calculate the average precision, recall and f1-scores.

![precision-recall](https://github.com/philipoedi/Color-Spaces-and-Conv-Net/blob/master/plots/precision_recall.png?raw=true)
![f1](https://github.com/philipoedi/Color-Spaces-and-Conv-Net/blob/master/plots/f1_scores.png?raw=true)


