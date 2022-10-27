#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:49:46 2022

@author: annamatsulevits
"""

import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np



# loading in image data:
    
base_dir = '/Users/annamatsulevits/Documents/GitHub/classification_DL_X-Ray/X_Ray_Classification/classification-challenge/classification-challenge-starter/Covid19-dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_dir_covid19 = os.path.join(train_dir, 'Covid')
train_dir_normal = os.path.join(train_dir, 'Normal')
train_dir_pneu = os.path.join(train_dir, 'Pneumonia')

test_dir_covid19 = os.path.join(test_dir, 'Covid')
test_dir_normal = os.path.join(test_dir, 'Normal')
test_dir_pneu = os.path.join(test_dir, 'Pneumonia')

train_covid19_images = os.listdir(train_dir_covid19)
train_normal_images = os.listdir(train_dir_normal)
train_pneu_images = os.listdir(train_dir_pneu)

test_covid19_images = os.listdir(test_dir_covid19)
test_normal_images = os.listdir(test_dir_normal)
test_pneu_images = os.listdir(test_dir_pneu)


BATCH_SIZE = 32
### IMAGE AUGMENTATION

training_data_generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05)

training_iterator = training_data_generator.flow_from_directory(train_dir, class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)

training_iterator.image_shape

print("\nLoading validation data...")

#1) Create validation_data_generator, an ImageDataGenerator that just performs pixel normalization:

validation_data_generator = ImageDataGenerator(
        rescale=1./255)

#2) Use validation_data_generator.flow_from_directory(...) to load the validation data from the 'data/test' folder:

validation_iterator = validation_data_generator.flow_from_directory(test_dir ,class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)



print("\nBuilding model...")

#Rebuilds our model from the previous exercise, with convolutional and max pooling layers:

model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=(256, 256, 1)))

model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(5, 5), strides=(5,5)))

model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(3,activation="softmax"))

model.summary()


print("\nCompiling model...")

#3) Compile the model with an Adam optimizer, Categorical Cross Entropy Loss, and Accuracy and AUC metrics:

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

print("\nTraining model...")

#4) Use model.fit(...) to train and validate our model for 5 epochs:

history = model.fit(
       training_iterator,
       steps_per_epoch=training_iterator.samples/BATCH_SIZE,
       epochs=10,
       validation_data=validation_iterator,
       validation_steps=validation_iterator.samples/BATCH_SIZE)



history.history.keys()

# Plot Training and Validation Loss


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Training and Validation Loss Trend')
plt.xlabel('Epochs')



##### codeacademy proposal
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
plt.show()
 

### EVENTUALLY ADD A CLASSIFICATION MATRIX AND A CONFUSION MATRIX

test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = np.math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())

from sklearn.metrics import classification_report

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(true_classes,predicted_classes)
print(cm)
