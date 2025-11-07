# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 23:03:50 2025

@author: kai-s
"""

"NC-DCNN == NON CONFORMANCE - DEEP CONVOLUTIONAL NEURAL NETWORK"

"IMPORTS"
"----------------------------------------------------------------------------"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras.utils import image_dataset_from_directory
import pickle

"----------------------------------------------------------------------------"

"Reproducability & Global Params"
np.random.seed(23)
tf.keras.utils.set_random_seed(23)
#tf.keras.mixed_precision.set_global_policy("mixed_float16")


"Step  1: Data Processing"
"----------------------------------------------------------------------------"
#NOTE: The data has already been split into the following categories: 
    #Training - 1942 images (66.7%)
    #Validation - 431 images (14.8%)
    #Testing - 539 images (18.5%)
        # 2912 images total

"DATA GENERATORS" #reading images off disk

train_generator = image_dataset_from_directory(
    directory = 'Project 2 Data/Data/train',
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (500, 500),
    interpolation = 'bilinear',
    batch_size = 32,
    color_mode = 'rgb', #denotes 3 channels (RGB)
    shuffle = True,
    verbose = True
    )

valid_generator = image_dataset_from_directory(
    directory = 'Project 2 Data/Data/valid',
    labels = 'inferred',
    label_mode = 'categorical',
    image_size = (500, 500),
    interpolation = 'bilinear',
    batch_size = 32,
    color_mode = 'rgb', #denotes 3 channels (RGB)
    shuffle = False,
    verbose = True
    )

"DATA AUGMENTATION" #applying random transformations onto images to improve generalization
#NOTE: By inspection, there is already augmentation applied to the datasets, so I made mine minimal

train_augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255), #TRYING THIS OUT 
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])

valid_augmentation_pipeline = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    ])

"APPLYING AUGMENTATIONS"
#lambda allows us to make an inline function
#tf.data.AUTOTUNE will automatically run a diagnostic on my PC resources and return optimal # of threads to use 
#.cache() will save data after first EPOCH to read directly from RAM and not disk on next EPOCH and beyond 
#.prefetch() starts prepping the next batch before the last is done training

train_data = train_generator.map(lambda x, y: (train_augmentation_pipeline(x, training = True), y), num_parallel_calls = tf.data.AUTOTUNE) 
train_data = train_data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

valid_data = valid_generator.map(lambda x, y: (valid_augmentation_pipeline(x, training = False), y), num_parallel_calls = tf.data.AUTOTUNE) 
valid_data = valid_data.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

#ADD TEST FOLDER LATER?

"----------------------------------------------------------------------------"

"Step  2: Neural Network Architecture Design"
"----------------------------------------------------------------------------"

"DCNN ARCHITECTURE"
"""model_v1 = models.Sequential([
    #INPUT
    Input(shape = (500, 500, 3)),
    #CONVOLUTIONAL BASE 
    layers.Conv2D(64, (6, 6), activation = 'relu'),
    layers.MaxPooling2D((4, 4)),
    layers.Conv2D(32, (4, 4), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (2, 2), activation = 'relu'),
    #FLATTEN
    layers.Flatten(),
    #FULLY CONNECTED LAYERS 
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation = 'softmax')
])

model_v1.summary()

"COMPILATION"
model_v1.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                loss = "categorical_crossentropy",
                metrics = ['accuracy']
                )"""

"EARLY STOP"
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "val_accuracy",
    patience = 3, #Changed to 3 after early stoppage at 1 in first run 
    restore_best_weights = True
    )

"RECORD HISTORY""""""
history = model_v1.fit(
    train_data,  
    validation_data = valid_data,
    epochs = 25, 
    batch_size = 32,
    callbacks = [early_stop],
    verbose = 1
    )"""

#MODEL
#L - CONVOLUTIONAL Conv2D Layers (# of filters, kernel size, stride) 
#L - MaxPooling2D 
#L - Flatten layer
#L - FULLY CONNECTED Dense Layers & Dropout Layers
#L - Last Dense Layer (3 Neurons)
#NOTE: Adjust FULLY CONNECTED and CONVOLUTIONAL layers to fine tune 

"----------------------------------------------------------------------------"

"Step  3: Hyperparameter Analysis"
"----------------------------------------------------------------------------"

#Can modify activation functions within convolutional & dense layers 
#Can Modify the number of filters & neurons within them
#Can modify loss function & Optimizers 

"DCNN ARCHITECTURE"
model_v2 = models.Sequential([
    #INPUT
    Input(shape = (500, 500, 3)),
    #CONVOLUTIONAL BASE 
    layers.Conv2D(16, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation = 'relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    #FLATTEN
    layers.GlobalAveragePooling2D(), #Trying this instead of Flatten()
    #FULLY CONNECTED LAYERS 
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation = 'softmax')
])

#increasing filter counts and standardized kernal size  
# added batch normalization
#Added GlobalAveragePooling instead of Flatten
#Added more convololutional layers

model_v2.summary()

"COMPILATION"
model_v2.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                loss = "categorical_crossentropy",
                metrics = ['accuracy']
                )

"RECORD HISTORY"
history = model_v2.fit(
    train_data,  
    validation_data = valid_data,
    epochs = 25, 
    batch_size = 32,
    callbacks = [early_stop], #use same early_stop as model_v1
    verbose = 1
    )


"----------------------------------------------------------------------------"

"Step  4: Model Evaluation"
"----------------------------------------------------------------------------"

#Set model for evaluation here: 
curr_model = model_v2

eval_loss, eval_accuracy = curr_model.evaluate(valid_data, verbose = 1)
print(f"Evaluation Accuracy: {eval_accuracy:.4f} | Evaluation Loss: {eval_loss:.4f}")

"PLOTTING ACCURACY AND LOSS FOR TRAINING AND VALIDATION DATA SETS "
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.title("CNN Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

"SAVING"
filename = "model_v2"
curr_model.save(f"models/{filename}_full.keras")
with open(f"models/{filename}_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
    
print(f"Successfully saved model as: models/{filename}_full.keras")
print(f"Successfully saved history as: models/{filename}_history.pkl")

"----------------------------------------------------------------------------"
