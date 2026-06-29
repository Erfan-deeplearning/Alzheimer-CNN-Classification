# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:29:05 2026

@author: mkh
"""

import pickle
import matplotlib.pyplot as plt

with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Accuracy
plt.figure(figsize=(8,5))
plt.plot(history["accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid()

plt.savefig("accuracy.png", dpi=300)
plt.show()

# Loss
plt.figure(figsize=(8,5))
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid()

plt.savefig("loss.png", dpi=300)
plt.show()

print("Plots saved successfully ✅")