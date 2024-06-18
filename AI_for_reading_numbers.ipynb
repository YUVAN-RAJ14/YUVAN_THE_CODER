import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import requests
from io import BytesIO

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape input data for CNN
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Build the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (if not already trained)
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Load pre-trained model (if already trained)
model = tf.keras.models.load_model('path_to_your_model')

# Function to preprocess image for prediction
def preprocess_image(image):
    # Resize image to 28x28 and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    # Convert image to numpy array and normalize
    image = np.array(image) / 255.0
    # Reshape to match input shape of the model
    image = image.reshape((-1, 28, 28, 1))
    return image

# Function to make predictions
def predict_digit(image_path, model):
    # Load image from path
    img = Image.open(image_path)
    # Preprocess image
    img_processed = preprocess_image(img)
    # Make prediction
    prediction = model.predict(img_processed)
    # Get predicted label
    predicted_label = np.argmax(prediction)
    return predicted_label

# Example usage:
# Replace 'image_path' with the path to your image file
image_path = input("Enter path for image")
predicted_digit = predict_digit(image_path, model)
print(f'Predicted digit: {predicted_digit}')
