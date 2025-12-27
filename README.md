**Fashion MNIST Image Classification** 

This project implements a deep learning model using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images in 10 categories, representing individual articles of clothing at low resolution (28 by 28 pixels).

**Dataset Overview**
The Fashion MNIST dataset is divided into:

Training Set: 60,000 images.

Test Set: 10,000 images.

Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

**Model Architecture**
The model is a Sequential neural network with the following layers:

Flatten: Converts the 28x28 image into a 1D array of 784 pixels.

Dense: 300 neurons with ReLU activation.

Dropout: Regularization layer to prevent overfitting.

Dense: 100 neurons with ReLU activation.

Dropout: Regularization layer.

Dense (Output): 10 neurons with Softmax activation for multi-class classification.

Total trainable parameters: 266,610.

**Training Configuration**
Loss Function: sparse_categorical_crossentropy.

Optimizer: Stochastic Gradient Descent (SGD).

Metrics: Accuracy.

Epochs: 30.

Validation: 5,000 samples taken from the training set.

**Performance**
The model achieves high accuracy on both training and validation sets. In the final epoch (30/30), the performance metrics were approximately:

Training Accuracy: ~87.8%

Validation Accuracy: ~88.2%

Validation Loss: ~0.43

**Setup and Usage**
Prerequisites:

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

Execution:

Load the dataset and normalize pixel values to a range of 0 to 1.

Define and compile the Sequential model.

Train the model using model.fit().

Evaluate the model and visualize training history.

Perform predictions on new data samples.
