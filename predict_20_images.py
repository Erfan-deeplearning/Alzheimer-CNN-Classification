# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:57:39 2026

@author: mkh
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# بارگذاری مدل
model = tf.keras.models.load_model("alzheimer_cnn_model.h5")

classes = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

folder = r"C:\Users\mkh\Desktop\Alzheimer_CNN_Project\dataset\OASIS_1_MRI_ALZHEIMER_DATA\Mild Impairment"

files = sorted(os.listdir(folder))

for file in files[:20]:
    img_path = os.path.join(folder, file)

    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    

    prediction = model.predict(img_array, verbose=0)

    pred_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"{file} -> {pred_class} ({confidence:.2f}%)")