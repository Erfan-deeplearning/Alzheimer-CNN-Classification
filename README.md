\# Alzheimer's Disease Classification Using CNN



\## Overview



This project implements a Convolutional Neural Network (CNN) for Alzheimer's disease classification using brain MRI images.



The model classifies MRI scans into four categories:



\* Mild Impairment

\* Moderate Impairment

\* No Impairment

\* Very Mild Impairment



\## Dataset



Dataset: OASIS MRI Alzheimer's Dataset



Total Images: 8000



Classes: 4



Images per Class: 2000



\## Technologies



\* Python

\* TensorFlow

\* Keras

\* NumPy

\* Matplotlib

\* Scikit-Learn

\* Seaborn



\## Model Architecture



\* Conv2D (32 filters)

\* MaxPooling2D

\* Conv2D (64 filters)

\* MaxPooling2D

\* Conv2D (128 filters)

\* MaxPooling2D

\* Flatten

\* Dense (128)

\* Dropout (0.3)

\* Dense (4, Softmax)



\## Training



Input Size: 128x128



Optimizer: Adam



Loss Function: Sparse Categorical Crossentropy



Epochs: 10



\## Results



The model successfully classifies MRI images into four Alzheimer's stages.



Generated outputs:



\* Accuracy Curve

\* Loss Curve

\* Confusion Matrix

\* Classification Report



\## Files



\* train.py

\* predict.py

\* evaluate.py

\* confusion\_matrix.py

\* plot\_history.py



\## Author



Erfan Ghaffari



M.Sc. in Artificial Intelligence



Semnan University



