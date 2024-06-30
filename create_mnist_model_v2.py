import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflowjs as tfjs
from tensorflow import keras

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels
test_labels = test_labels

train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

def create_model():
  model = tf.keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Display the model's architecture
model.summary()

loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Model accuracy: {:5.2f}%".format(100 * acc))

tfjs.converters.save_keras_model(model, "./model")
