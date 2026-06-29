# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 12:21:15 2026

@author: mkh
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# مسیر دیتاست
DATASET_PATH = r"C:\Users\mkh\Desktop\Alzheimer_CNN_Project\dataset\OASIS_1_MRI_ALZHEIMER_DATA"

# بارگذاری دیتاست
test_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

# بارگذاری مدل
model = tf.keras.models.load_model("alzheimer_cnn_model.h5")

# ارزیابی
loss, accuracy = model.evaluate(test_ds)

print(f"\nLoss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")