# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:13:28 2024

@author: dell 
"""

# Import libraries 

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Set parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_OF_CLASSES = 5

# Training and validation data directories
TRAINING_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/train"
VALIDATION_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/test"

# Image preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Training and validation data generators
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    target_size=IMG_SIZE
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
    target_size=IMG_SIZE
)

# Define MobileNet base model
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the MobileNet base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_OF_CLASSES, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Model summary
model.summary()

# Define callbacks for model checkpoint and learning rate reduction
callbacks_list = [
    ModelCheckpoint(filepath=".keras", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1.5e-5)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=validation_generator
)

# Plot training history
plt.figure(figsize=(15, 7))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig("MobileNet_train_val_loss.png")

plt.figure(figsize=(15, 5))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("MobileNet_train_val_accuracy.png")
plt.show()

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f'Test loss: {evaluation[0]} / Test accuracy: {evaluation[1]}')

# Generate predictions
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Generate true labels
y_true = validation_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

# Define class labels
class_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus', 'Ringworm']

# Plot confusion matrix using seaborn
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, linewidths=.9)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('MobileNet_confusion_matrix.png')

# Save the final model
model.save("MobileNet.h5")

# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv('MobileNet_training_history.csv', index=False)

# Save classification report to a text file
report = classification_report(validation_generator.classes, y_pred, target_names=class_names)
with open("MobileNet_classification_report.txt", "w") as text_file:
    text_file.write(report)

print("Classification report saved to 'MobileNet_classification_report.txt'")
