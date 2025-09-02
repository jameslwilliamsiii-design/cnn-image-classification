Fashion MNIST CNN Classifier

This project implements a convolutional neural network (CNN) in TensorFlow/Keras to classify grayscale images from the Fashion MNIST dataset. The dataset includes 70,000 images of clothing items spread across 10 categories.

Overview

The model takes 28x28 grayscale images and classifies them into one of the following categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. The training process includes normalization, reshaping of input data, and evaluation using a confusion matrix.

Model Architecture

The CNN architecture includes two convolutional and pooling layers, followed by a flattening layer and two dense layers:

Conv2D → ReLU activation → MaxPooling2D

Conv2D → ReLU activation → MaxPooling2D

Flatten → Dense(64, ReLU) → Dense(10, softmax)

The model is compiled with the Adam optimizer and trained using sparse categorical cross-entropy loss.

Training and Evaluation

The model is trained for 10 epochs with a batch size of 64 and a validation split of 20%. After training, it is evaluated on a separate test set to determine classification accuracy. A confusion matrix is generated to visualize classification performance across all classes.

Requirements

tensorflow

numpy

matplotlib

seaborn

scikit-learn

You can install these using:

pip install tensorflow numpy matplotlib seaborn scikit-learn

Running the Code

Make sure all dependencies are installed, then run the script:

python cnn_fashion_mnist.py


The output includes model training logs, evaluation metrics, and a confusion matrix plot.

License

This project is released under the MIT License.
