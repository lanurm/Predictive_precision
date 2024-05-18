import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define image dimensions
img_height = 64
img_width = 64

# Define the model architecture
def create_crop_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # Adjust to 3 output neurons for 3 categories
    return model

# Load and preprocess images and labels
def load_crop_data(base_dir):
    images = []
    labels = []
    classes = sorted(os.listdir(base_dir))  # Get a sorted list of subdirectory names
    for label, folder in enumerate(classes):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):  # Assuming images are in JPG format
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    img = cv2.resize(img, (img_height, img_width))
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels).reshape(-1, 1)


# Load all crop data
base_directory = './images'  # Adjust the directory path accordingly
all_images, all_labels = load_crop_data(base_directory)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
print(train_images.shape, val_images.shape, train_labels.shape, val_labels.shape)

# Create and compile the model
model = create_crop_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_images, val_labels)
#print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

# Save the trained model as an h5 file
model.save('crop_classification_model.h5')
