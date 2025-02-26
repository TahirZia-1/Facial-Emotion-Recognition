# Facial Expression Recognition with EfficientNetB2

This repository implements a facial expression recognition system using the EfficientNetB2 model from TensorFlow/Keras. The project classifies facial images into seven emotion categories (angry, disgust, fear, happy, neutral, sad, surprise) using a dataset with 28,821 training and 7,066 validation images. The trained model leverages transfer learning, data augmentation, and class weighting to handle imbalanced data, and it’s saved for future inference.

## Overview

The goal is to classify facial expressions from RGB images (96x96) into seven emotion classes. The dataset is preprocessed with data augmentation and one-hot encoding, and the model is trained with class weights to address class imbalance. The EfficientNetB2 backbone is fine-tuned with additional layers for improved performance.

### Key Features
- **Dataset**: 28,821 training and 7,066 validation images across 7 classes.
- **Model**: EfficientNetB2 with custom Conv2D, GlobalAveragePooling2D, Dense, and Dropout layers.
- **Preprocessing**: Resizing, random flips, rotations, and zooms for augmentation.
- **Class Balancing**: Class weights to mitigate imbalance (e.g., "disgust" has fewer samples).
- **Evaluation**: Accuracy, precision, and recall metrics.

## Implementation

### Dataset
- **Structure**: 
  - `images/train/`: 28,821 images (7 subfolders for each class).
  - `images/validation/`: 7,066 images (7 subfolders).
- **Classes**: Angry (0), Disgust (1), Fear (2), Happy (3), Neutral (4), Sad (5), Surprise (6).
- **Preprocessing**: Images resized to 96x96, normalized, and augmented.

### Model Architecture
- **Input**: 96x96x3 (RGB images).
- **Backbone**: EfficientNetB2 (pre-trained, no top).
- **Custom Layers**:
  - Conv2D: 128 filters, 3x3 kernel, ReLU activation.
  - GlobalAveragePooling2D.
  - Dense: 128 units, ReLU activation.
  - Dropout: 0.3.
  - Dense: 7 units, softmax activation (output).
- **Parameters**: ~9.41M total, ~9.34M trainable.

### Training
- **Optimizer**: Adam (learning rate: 0.001).
- **Loss**: Categorical Crossentropy.
- **Metrics**: Accuracy, Precision, Recall.
- **Epochs**: 8.
- **Batch Size**: 64.
- **Class Weights**: Computed to balance class distribution (e.g., `{0: 1.79, 1: 16.43, ...}`).

### Results
- **Training**:
  - Best Accuracy: 85.66% (Epoch 1).
  - Precision: ~87%, Recall: ~83%.
- **Validation**:
  - Accuracy: ~1.58% (Epoch 8).
  - Loss: 2.1885, Precision: 0.0, Recall: 0.0 (indicating overfitting).
- **Note**: Validation performance suggests overfitting; further tuning is needed.

*Class Distribution Plot*:  
![Class Distribution](images/class_distribution.jpg)  
(Saved from `facial_expression_recognition.ipynb` as `images/class_distribution.png`.)

## Installation

### Prerequisites
- Python 3.x
- Required libraries:
  - TensorFlow (`pip install tensorflow`)
  - NumPy (`pip install numpy`)
  - Pandas (`pip install pandas`)
  - Matplotlib (`pip install matplotlib`)
  - Seaborn (`pip install seaborn`)
  - OpenCV (`pip install opencv-python`)
  - Scikit-learn (`pip install scikit-learn`)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TahirZia-1/facial-expression-recognition.git
   cd facial-expression-recognition
