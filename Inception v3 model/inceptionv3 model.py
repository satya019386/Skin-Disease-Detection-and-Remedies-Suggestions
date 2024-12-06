# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:13:28 2024

@author: dell 
"""

#inception v3 model

# import libraries 

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from urllib.request import urlopen
import matplotlib.image as mpimg
import random
import zipfile, os
import warnings
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import  ResNet50V2
#########################################################################
""" Inspect the dataset                             """
#########################################################################




#########################################################################
""" Image preprocessing and augmentation              """
#########################################################################

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PATIENCE = 8
NUM_EPOCHS = 15
NUM_OF_CLASSES = 5
INIT_LR = 0.01

#########################################################################
""" Training data source                           """
#########################################################################

TRAINING_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/train"

train_datagen = ImageDataGenerator(
    rescale = 1/255.0,
    #rotation_range=40,                  # Randomly rotate images by up to 40 degrees
    #width_shift_range=0.2,              # Randomly shift images horizontally by up to 20%
    
    #height_shift_range=0.2,             # Randomly shift images vertically by up to 20%
    #shear_range=0.2,                    # Shear transformations
    #zoom_range=0.2,                     # Randomly zoom in/out
    #horizontal_flip=True,               # Randomly flip images horizontally
   #vertical_flip=True,                 # Randomly flip images vertically
    #brightness_range=[0.5, 1.5],        # Adjust brightness
    #channel_shift_range=50.0,           # Adjust color channel values
    fill_mode='nearest'                 # Fill in missing pixels using the nearest available pixel
    
) 

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, 
    batch_size = BATCH_SIZE,
    shuffle=True,
    class_mode = 'categorical',
    target_size = (224, 224)
)

#########################################################################
""" Validation data source                         """
#########################################################################


VALIDATION_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/test"
validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
    target_size = (224, 224)
)

#########################################################################
""" Handling class imbalance using class weight approach

wj=Totat_number_of_Samples / (number of classes* number of samples in jth classs)
"""
#########################################################################
class_weight = class_weight.compute_class_weight(
          class_weight = 'balanced',
          classes = np.unique(train_generator.classes), 
          y=train_generator.classes)

class_weight = dict(zip(np.unique(train_generator.classes), class_weight))

#########################################################################
""" Hyperparameter Tuning and Early Stopping             """
#########################################################################

save_path= "InceptionV3.h5"
checkpoint_path = ".keras"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks_list = [
    
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
       monitor='val_loss', 
       factor=0.2,
       patience=8, 
       min_lr=1.5e-5
   ),
]

#########################################################################
""" Model Building                               """
#########################################################################
# Load the MobileNet model (you can specify the input shape and weights)
base_model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')


for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the MobileNet base
x = base_model.output
#x = Flatten()(x)  # Flatten layer
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # Dense layer with 1024 units
x= Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_OF_CLASSES , activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)




#########################################################################
""" Compiling the model                               """
#########################################################################

model.compile(
    optimizer = 'adam', 
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
)

#########################################################################
""" Model Summary                                """
#########################################################################


model.summary()

#########################################################################
""" Training the model                               """
#########################################################################

history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=validation_generator,
    class_weight=class_weight
)

#####################################################################


#########################################################################
""" Plotting the accuracy and loss graph                 """
#########################################################################

plt.figure(4, figsize=(15, 7))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig("dense_train_val_loss.png")

plt.figure(5, figsize=(15, 5))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("dense_train_val_accuracy.png")
plt.show()



#########################################################################
""" Evaluate the Model                           """
#########################################################################

vgg_eval_Score = model.evaluate(validation_generator)
print(f'Test loss: {vgg_eval_Score[0]} / Test accuracy: {vgg_eval_Score[1]}')

###############################################################################
""" Plot few results """

filenames = validation_generator.filenames
nb_samples = len(filenames)
Y_pred = model.predict(validation_generator, steps = nb_samples)
y_pred = np.argmax(Y_pred, axis=1)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(validation_generator.classes, y_pred)
print('accuracy_score: ')
print(accuracy)


print('Classification Report')
target_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus','Ringworm']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# Generate predictions
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Generate true labels
y_true = validation_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

# Define class labels
class_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus','Ringworm']

# Plot confusion matrix using seaborn
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names,linewidths=.9)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
fig = plt.gcf()
fig.savefig('confusion_matrix.png')

plt.savefig('confusion_matrix.png')

######################################################################
""" Save the final Model                         """   
model.save("InceptionV3.h5")
######################################################################

######################################################################
""" Save the history                         """   
import pandas as pd
# Save the training history to a CSV file
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
######################################################################
report = classification_report(validation_generator.classes, y_pred, target_names=target_names)

with open("classification_report.txt", "w") as text_file:
    text_file.write(report)

print("Classification report saved to 'classification_report.txt'")

























