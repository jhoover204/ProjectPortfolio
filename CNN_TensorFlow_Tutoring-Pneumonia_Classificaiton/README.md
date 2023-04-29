## Chest X-Ray Pneumonia Classification using Convolutional Neural Networks

This project uses a convolutional neural network (CNN) to classify chest x-ray images as either showing signs of pneumonia or not. The CNN was developed to demonstrate how to perform this type of classification to a tutoring client. The model was developed using the provided training and testing datasets (pre-split) provided by the client. No tuning was performed on the model, as the goal of the project was to demonstrate the general process of using a CNN for image classification.

## Dataset

The dataset used in this project consists of labeled chest x-ray images. The images have already been split into training and testing sets by the client, so no additional splitting was performed. The dataset consists of:

5216 training images
624 testing images

## Requirements

The project requires Python 3 and the following libraries:

TensorFlow
NumPy
Matplotlib
Scikit-learn
Pandas
OS
CV2

You can install these dependencies using pip


## Model Architecture

The CNN model used for this classification task has the following architecture:

Model: "sequential_12"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_55 (Conv2D)           (None, 148, 148, 32)      320       
                                                                 
max_pooling2d_55 (MaxPooling (None, 74, 74, 32)        0         
2D)                                                             
                                                                 
conv2d_56 (Conv2D)           (None, 72, 72, 64)        18496     
                                                                 
max_pooling2d_56 (MaxPooling (None, 36, 36, 64)        0         
2D)                                                             
                                                                 
conv2d_57 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
max_pooling2d_57 (MaxPooling (None, 17, 17, 128)       0         
2D)                                                             
                                                                 
conv2d_58 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                 
max_pooling2d_58 (MaxPooling (None, 7, 7, 128)         0         
2D)                                                             
                                                                 
conv2d_59 (Conv2D)           (None, 5, 5, 128)         147584    
                                                                 
max_pooling2d_59 (MaxPooling (None, 2, 2, 128)         0         
2D)                                                             
                                                                 
dropout_12 (Dropout)         (None, 2, 2, 128)         0         
                                                                 
flatten_12 (Flatten)         (None, 512)               0         
                                                                 
dense_24 (Dense)             (None, 512)               262656    
                                                                 
dense_25 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 651,009
Trainable params: 651,009
Non-trainable params: 0
_________________________________________________________________


## Usage

The provided model can be used to make predictions on new chest x-ray images. However, it should be noted that no tuning was performed and the model was designed simply to demonstrate how a CNN can be used for this type of classification task. If the client wishes to improve the performance of the model, they should consider performing their own tuning.