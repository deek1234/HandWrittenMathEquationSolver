Handwritten Equation Solver

Overview

The Handwritten Equation Solver is an advanced computer vision application that recognizes and solves handwritten mathematical equations. This innovative tool interprets and evaluates simple equations containing basic arithmetic operations (+, -, *, /) and digits (0-9) from images.

Features

Recognizes handwritten mathematical symbols and digits
Parses and solves simple arithmetic equations
Processes various image formats (JPEG, PNG)
Provides visual feedback with bounding boxes and recognized symbols
Outputs the parsed digital equation and its calculated result

Technical Details

Model Architecture

The project uses a Sequential Convolutional Neural Network (CNN) implemented with TensorFlow and Keras. The model architecture includes:
Three convolutional layers with ReLU activation and max pooling
Dropout for regularization
Two fully connected layers
Softmax output layer for 14-class classification

Algorithms and Techniques

CNN: For feature extraction and classification of handwritten symbols
Data Augmentation: Using Keras' ImageDataGenerator to enhance model generalization
Learning Rate Scheduling: Custom step decay function for optimizing model convergence
Image Preprocessing: Grayscale conversion, Otsu's thresholding, and resizing to 32x32 pixels
Contour Detection and Sorting: Using OpenCV and imutils for symbol identification and sequencing

Performance

Overall accuracy: 96%
F1 scores ranging from 0.97 to 1.00 across all classes
Consistent macro averages of 0.96 for precision, recall, and F1-score

Dataset

The project uses a custom dataset of 7,600 handwritten math symbol images, including digits 0-9 and operators +, -, *, /. The dataset is split into 80% training (6,080 images) and 20% testing (1,520 images).

Requirements

TensorFlow
Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
PIL
imutils
