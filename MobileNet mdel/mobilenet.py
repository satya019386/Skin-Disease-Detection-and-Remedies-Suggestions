# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:13:28 2024

@author: dell 
"""

# Import libraries 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

#########################################################################
""" Image preprocessing and augmentation              """
#########################################################################

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_OF_CLASSES = 5

#########################################################################
""" Training data source                           """
#########################################################################

TRAINING_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/train"

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    fill_mode='nearest'  # Only rescale images for now
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    target_size=(224, 224)
)

#########################################################################
""" Validation data source                         """
#########################################################################

VALIDATION_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/test"

validation_datagen = ImageDataGenerator(rescale=1/255.0)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
    target_size=(224, 224)
)

#########################################################################
""" Handling class imbalance using class weight approach            """
#########################################################################

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))

#########################################################################
""" Model Building                               """
#########################################################################

# Load the MobileNet model (you can specify the input shape and weights)
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

model = Model(inputs=base_model.input, outputs=predictions)

#########################################################################
""" Compiling the model                               """
#########################################################################

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

#########################################################################
""" Training the model                               """
#########################################################################

callbacks_list = [
    ModelCheckpoint(filepath=".keras", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1.5e-5)
]

history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=validation_generator,
    class_weight=class_weights
)

#########################################################################
""" Plotting the accuracy and loss graph                 """
#########################################################################

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

#########################################################################
""" Evaluate the Model                           """
#########################################################################

evaluation = model.evaluate(validation_generator)
print(f'Test loss: {evaluation[0]} / Test accuracy: {evaluation[1]}')

#########################################################################
""" Generate predictions and save results                           """
#########################################################################

Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Classification Report')
target_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus', 'Ringworm']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

conf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=target_names, yticklabels=target_names, linewidths=.9)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('MobileNet_confusion_matrix.png')

model.save("MobileNet.h5")

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.to_csv('MobileNet_training_history.csv', index=False)

report = classification_report(validation_generator.classes, y_pred, target_names=target_names)
with open("MobileNet_classification_report.txt", "w") as text_file:
    text_file.write(report)

print("Classification report saved to 'MobileNet_classification_report.txt'")
