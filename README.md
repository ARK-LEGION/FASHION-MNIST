# Fashion MNIST Image Classification

A deep learning project implemented in a Jupyter Notebook using TensorFlow and Keras to classify clothing items from the Fashion MNIST dataset.

## Project Overview
This project builds a Sequential neural network to categorize grayscale images of clothing into one of ten distinct classes. The model utilizes dense layers with ReLU activation and Dropout layers for regularization to achieve robust classification performance.

## Dataset
The project uses the **Fashion MNIST dataset**, which is built into Keras.
* **Data Split**: 60,000 training images and 10,000 testing images.
* **Image Dimensions**: 28x28 pixels (grayscale).
* **Preprocessing**: Input pixel values (0-255) are normalized to a range of 0.0 to 1.0 to ensure stable training.
* **Categories**: 
    1. T-shirt/top
    2. Trouser
    3. Pullover
    4. Dress
    5. Coat
    6. Sandal
    7. Shirt
    8. Sneaker
    9. Bag
    10. Ankle boot

## Model Architecture
The model is defined as a `Sequential` network with the following layers:
1.  **Flatten**: Converts the 2D 28x28 image into a 1D array of 784 pixels.
2.  **Dense**: 300 neurons with ReLU activation.
3.  **Dropout**: Regularization layer to prevent overfitting.
4.  **Dense**: 100 neurons with ReLU activation.
5.  **Dropout**: Additional regularization layer.
6.  **Dense (Output)**: 10 neurons with Softmax activation for multi-class probability distribution.

**Total Trainable Parameters**: 266,610.

## Training Configuration
* **Optimizer**: Stochastic Gradient Descent (SGD).
* **Loss Function**: `sparse_categorical_crossentropy`.
* **Metric**: Accuracy.
* **Epochs**: 30.
* **Validation**: 5,000 samples reserved from the training set.

## Results
After 30 epochs of training, the model achieved the following approximate metrics:
* **Training Accuracy**: ~87.8%
* **Validation Accuracy**: ~88.2%
* **Validation Loss**: ~0.43

## Dependencies
The following libraries are required to run the notebook:
* `tensorflow`
* `keras`
* `numpy`
* `matplotlib`

## Usage
1.  Open `fashion_mnist.ipynb` in a Jupyter environment.
2.  Run the initialization cells to load and normalize the dataset.
3.  Execute the model definition and training cells.
4.  View the generated plots to see training history (loss and accuracy) and visual predictions.
