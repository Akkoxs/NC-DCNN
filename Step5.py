# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 06:00:14 2025

@author: kai-s
"""

"IMPORTS"
"----------------------------------------------------------------------------"
import tensorflow as tf 
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

"----------------------------------------------------------------------------"

"Step  5: Model Testing"
"----------------------------------------------------------------------------"

test_direc = 'Project 2 Data/Data/test'

baseline_model = tf.keras.models.load_model('models/model_v0_secundus_full.keras')
variant_model = tf.keras.models.load_model('models/model_v0_tertius_full.keras')

#function for processing image
def process_image(img_path):
    img = image.load_img(img_path, target_size = (500, 500)) #load          
    img_array = image.img_to_array(img) #convert to array                              
    img_array = img_array / 255.0 #normalize image [0, 1]
    img_array = np.expand_dims(img_array, axis = 0)
    return img_array

detected_classes = sorted(os.listdir(test_direc))
print("Detected classes:", detected_classes)

baseline_results = []
variant_results = []

# Loop through all folders and images and predict/store 
for class_name in detected_classes:
    class_folder = os.path.join(test_direc, class_name)

    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)

        img_array = process_image(img_path)
        #predict using models 
        baseline_pred = baseline_model.predict(img_array, verbose = 1)
        variant_pred = variant_model.predict(img_array, verbose = 1)
        #index for class (for ref)
        baseline_pred_class_index = np.argmax(baseline_pred)                   
        variant_pred_class_index = np.argmax(variant_pred)          
        #class name predictions
        baseline_pred_class_name = detected_classes[baseline_pred_class_index]    
        variant_pred_class_name = detected_classes[variant_pred_class_index]  
        #calculating confidence, max because softmax     
        baseline_confidence = np.max(baseline_pred)                           
        variant_confidence = np.max(variant_pred)                              

        baseline_results.append({
            "image": img_file,
            "true_label": class_name,
            "predicted_label": baseline_pred_class_name,
            "confidence": float(baseline_confidence)
        })
            
        variant_results.append({
            "image": img_file,
            "true_label": class_name,
            "predicted_label": variant_pred_class_name,
            "confidence": float(variant_confidence)    
        })

"METRICS"
"----------------------------------------------------------------------------"

def per_class_accuracy(results, class_names, model_name):
    print(f"PER-CLASS ACCURACY - {model_name}")
    for cls in class_names:
        cls_results = [r for r in results if r["true_label"] == cls]
        correct = sum(1 for r in cls_results if r["predicted_label"] == r["true_label"])
        total_cls = len(cls_results)
        acc = correct / total_cls if total_cls > 0 else 0
        print(f"  {cls}: {correct}/{total_cls} correct, Accuracy = {acc:.4f}")
    print("\n")


baseline_corr_preds = sum(1 for r in baseline_results if r["true_label"] == r["predicted_label"])
variant_corr_preds = sum(1 for r in variant_results if r["true_label"] == r["predicted_label"])
total = len(baseline_results)
baseline_accuracy = baseline_corr_preds/total
variant_accuracy = variant_corr_preds/total
    
print(f"Total Images: {total}")
print("BASELINE MODEL - METRICS")
print(f"Correct Predictions: {baseline_corr_preds}")
print(f"Accuracy: {baseline_accuracy:.4f}\n")
per_class_accuracy(baseline_results, detected_classes, "Baseline Model")

print("VARIANT MODEL - METRICS")
print(f"Correct Predictions: {variant_corr_preds}")
print(f"Accuracy: {variant_accuracy:.4f}\n")
per_class_accuracy(variant_results, detected_classes, "Variant Model")


"TESTING & PLOTTING FINAL IMAGES"
"----------------------------------------------------------------------------"

crack_test_direc = 'Project 2 Data/Data/test/crack/test_crack.jpg'
missinghead_test_direc = 'Project 2 Data/Data/test/missing-head/test_missinghead.jpg'
paintoff_test_direc = 'Project 2 Data/Data/test/paint-off/test_paintoff.jpg'

final_test_images = [
    ('crack', crack_test_direc),
    ('missing-head', missinghead_test_direc),
    ('paint-off', paintoff_test_direc)
    ]

def plot_final_predictions(model, model_name, class_names):
    plt.figure(figsize = (12, 6))
    for i, (label, path) in enumerate(final_test_images):
        img_array = process_image(path)
        pred = model.predict(img_array, verbose = 1)[0]
        predicted_class = class_names[np.argmax(pred)]
        confidences = {cls: float(pred[j]) for j, cls in enumerate(class_names)}

        img = image.load_img(path)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")

        conf_text = "\n".join([f"{cls}: {confidences[cls]:.2f}" for cls in class_names])
        plt.title(f"{model_name}\nTrue: {label}\nPred: {predicted_class}\n\n{conf_text}",
                  fontsize = 10)

    plt.tight_layout()
    plt.show()

#plot
plot_final_predictions(baseline_model, "Baseline Model", detected_classes)
plot_final_predictions(variant_model, "Variant Model", detected_classes)


#Test the model that was trained 

#INSTRUCTIONS
#Data processing of test image to convert image to same format as input of model 
    #use image package by Keras 
    #load image 
    #convert image to array 
    #normalize it by dividing image by 255 
    
#find maximum propability from models prediction 
#final prediction of image is expected to look like fig3


"----------------------------------------------------------------------------"
