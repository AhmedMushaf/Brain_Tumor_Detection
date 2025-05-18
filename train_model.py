import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

# Image size and labels
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Load and preprocess data
X, Y = [], []

# Load training data
for label in labels:
    folder = os.path.join('dataset/Training', label)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        Y.append(label)

# Load test data
for label in labels:
    folder = os.path.join('dataset/Testing', label)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.resize(img, (image_size, image_size))
        X.append(img)
        Y.append(label)

# Normalize and prepare labels
X = np.array(X) / 255.0
Y = to_categorical([labels.index(y) for y in Y])
X, Y = shuffle(X, Y, random_state=101)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=101)

# Build model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=32)

# Save model
model.save("brain_tumor_model.h5")

# Save label list
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

# Plot and save accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.close()

# Plot and save loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.close()