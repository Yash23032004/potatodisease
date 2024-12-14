import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 30

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/yashchaudhary/potatodisease/training/Potatodataset",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Class names and distribution
class_names = dataset.class_names
class_counts = np.bincount([label.numpy() for image_batch, label_batch in dataset for label in label_batch])
print("Class distribution:", dict(zip(class_names, class_counts)))

# Adjust class weights based on the class distribution
class_weights = {0: 1.0, 1: 3.0, 2: 1.5}  # Adjust this based on actual class imbalance

# Train/validation/test split
train_size = 0.8
train_ds = dataset.take(int(len(dataset) * train_size))
val_ds = dataset.skip(int(len(dataset) * train_size)).take(int(len(dataset) * 0.1))
test_ds = dataset.skip(int(len(dataset) * (train_size + 0.1)))

# Caching and prefetching datasets for faster I/O
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Data preprocessing (Resizing and rescaling)
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0 / 255)
])

# Data augmentation (disabled for debugging)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Model architecture (simplified)
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,  # Data augmentation enabled
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate for better convergence
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Early stopping setup (stop if validation loss doesn't improve for 5 epochs)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    class_weight=class_weights,  # Apply class weights to handle class imbalance
    callbacks=[early_stopping]   # Early stopping to prevent overfitting
)

# Evaluate the model on the test dataset
score = model.evaluate(test_ds)
print("Test loss, Test accuracy:", score)

# Plot training/validation accuracy and loss
epochs_completed = len(history.history['accuracy'])  # Number of epochs completed during training

# Plot accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs_completed), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(epochs_completed), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(range(epochs_completed), history.history['loss'], label='Training Loss')
plt.plot(range(epochs_completed), history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Predict on a batch of test images to visually verify predictions
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        plt.axis("off")

plt.show()

# Saving the model
models_dir = "../models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

existing_versions = [
    int(i.split('.')[0]) for i in os.listdir(models_dir)
    if i.split('.')[0].isdigit()  # Ensure we are only considering numeric filenames
]

model_version = max(existing_versions + [0]) + 1
model.save(f"{models_dir}/{model_version}.keras")

model.save("../potatoes1.keras")