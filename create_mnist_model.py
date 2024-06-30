import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

def create_cnn_model():
    model = tf.keras.Sequential([
        keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    return model


# Create a model instance
model = create_cnn_model()

# Train the model
model.fit(train_images, train_labels, epochs=20, batch_size=32)

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Model accuracy: {:5.2f}%".format(100 * acc))

# Save the model for TensorFlow.js
tfjs.converters.save_keras_model(model, "./model")
