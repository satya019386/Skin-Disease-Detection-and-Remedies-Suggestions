import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_OF_CLASSES = 5
INIT_LR = 0.001

# Define data directories
TRAINING_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/train"
VALIDATION_DIR = "C:/Users/dell/Desktop/AnshuRani Project/Dataset/test"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    target_size=IMG_SIZE
)

validation_datagen = ImageDataGenerator(rescale=1/255.0)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical',
    target_size=IMG_SIZE
)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_generator.classes),
    train_generator.classes
)

class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Define callbacks
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='resnet50_model.h5', 
        monitor='val_loss', 
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=8, 
        min_lr=1e-5
    ),
]

# Load ResNet50 base model
base_model = tf.keras.applications.ResNet50V2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_OF_CLASSES, activation='softmax')(x)

# Compile the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(lr=INIT_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Plot training history
plt.figure(figsize=(15, 7))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig("resnet50_train_val_loss.png")

plt.figure(figsize=(15, 5))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("resnet50_train_val_accuracy.png")
plt.show()

# Evaluate the model
eval_score = model.evaluate(validation_generator)
print(f'Test loss: {eval_score[0]} / Test accuracy: {eval_score[1]}')

# Generate predictions
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Generate true labels
y_true = validation_generator.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Define class labels
class_names = ['Acne', 'Dermatitis', 'Healthy', 'Lupus', 'Ringworm']

# Plot confusion matrix
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, linewidths=.9)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('resnet50_confusion_matrix.png')

# Save the model
model.save("resnet50_model.h5")

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('resnet50_training_history.csv', index=False)

# Save classification report
report = classification_report(
    validation_generator.classes, y_pred, target_names=class_names
)
with open("resnet50_classification_report.txt", "w") as text_file:
    text_file.write(report)

print("Classification report saved to 'resnet50_classification_report.txt'")


