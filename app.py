# Fashion MNIST CNN Classifier

from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize one image per class
plt.figure(figsize=(12, 6))
for class_index in range(10):
    idx = np.where(train_labels == class_index)[0][0]
    plt.subplot(2, 5, class_index + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[class_index])
    plt.axis('off')
plt.suptitle('Sample Images from Each Class', fontsize=16)
plt.tight_layout()
plt.show()

# Reshape and normalize
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(train_images, train_labels, epochs=10,
                    batch_size=64, validation_split=0.2, verbose=2)

# Evaluate performance
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Predict and analyze
pred_probs = model.predict(test_images)
pred_labels = tf.argmax(pred_probs, axis=1)
cm = confusion_matrix(test_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.tight_layout()
plt.show()
