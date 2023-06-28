import tensorflow as tf
from tensorflow import keras

# Load and preprocess the iris dataset
iris = keras.datasets.iris
(train_images, train_labels), (test_images, test_labels) = iris.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define and compile the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100}%")
