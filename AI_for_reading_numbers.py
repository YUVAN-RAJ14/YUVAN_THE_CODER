import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import requests
    from PIL import Image
    from io import BytesIO
    import urlib.request 
except ImportError:
    print("Some libraries not found. Installing dependencies...")
    install('tensorflow')
    install('numpy')
    install('matplotlib')
    install('Pillow')
    install('io')
    install('requests')
    install('urlib')
    from PIL import Image
    from io import BytesIO
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import urlib.request
    import requests 
    import numpy as np
    
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image) / 255.0
    image = image.reshape((-1, 28, 28, 1))
    return image

def predict_digit(image_path, model):
    img = Image.open(image_path)
    img_processed = preprocess_image(img)
    prediction = model.predict(img_processed)
    predicted_label = np.argmax(prediction)
    return predicted_label
 
image_path = input("Enter path for image")
predicted_digit = predict_digit(image_path, model)
print(f'Predicted digit: {predicted_digit}')
