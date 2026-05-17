import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(150, 150),
    batch_size=10,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(150, 150),
    batch_size=10,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150,150,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model

# model.save("E:/ML_Model/model.keras")
model.save("model.keras")

print("Model saved successfully!")