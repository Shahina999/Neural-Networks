# Neural Networks with MNIST Dataset

This repository contains two neural network models—Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN)—implemented using the MNIST dataset of handwritten digits. The MNIST dataset is commonly used for benchmarking classification models in the field of machine learning and deep learning.

## Files in the Repository

- `CNN_Model.ipynb`: This file contains the implementation of a **Convolutional Neural Network (CNN)** for digit classification using the MNIST dataset.
- `ANN_Model.ipynb`: This file contains the implementation of an **Artificial Neural Network (ANN)** for digit classification using the MNIST dataset.

## Dataset

The **MNIST dataset** is a collection of 70,000 images of handwritten digits from 0 to 9, with each image being 28x28 pixels in grayscale. It is pre-loaded in popular machine learning libraries such as TensorFlow and PyTorch, making it easily accessible.

## Model Descriptions

### 1. CNN Model
The Convolutional Neural Network (CNN) is built using the following architecture:
- **Input Layer**: 28x28 pixel grayscale images
- **Convolutional Layers**: Extract spatial features from images
- **Pooling Layers**: Downsample feature maps
- **Fully Connected Layer**: For classification based on extracted features
- **Output Layer**: 10 units (one for each digit class)

The CNN is optimized using an appropriate loss function and optimizer, and the performance is measured in terms of accuracy.

### 2. ANN Model
The Artificial Neural Network (ANN) follows a simpler architecture:
- **Input Layer**: 784 input neurons (28x28 flattened image)
- **Hidden Layers**: One or more dense layers with activation functions
- **Output Layer**: 10 neurons corresponding to digit classes (0-9)

The ANN is trained using a similar loss function, and its performance is also evaluated in terms of classification accuracy.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/Shahina999/Neural-Networks.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks to train and evaluate the models on the MNIST dataset.

   ```bash
   jupyter notebook CNN_Model.ipynb
   jupyter notebook ANN_Model.ipynb
   ```

## Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
