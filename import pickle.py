# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:04:17 2026

@author: mkh
"""

import pickle
import matplotlib.pyplot as plt

# بارگذاری history
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# نمودار Accuracy
plt.figure(figsize=(10,5))
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# نمودار Loss
plt.figure(figsize=(10,5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()