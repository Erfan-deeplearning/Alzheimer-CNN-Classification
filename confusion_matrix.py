# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:22:31 2026

@author: mkh
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# بارگذاری مدل
model = tf.keras.models.load_model("alzheimer_cnn_model.h5")

# بارگذاری داده‌های اعتبارسنجی
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/OASIS_1_MRI_ALZHEIMER_DATA",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128,128),
    batch_size=32,
    shuffle=True
)
import os

root = "dataset/OASIS_1_MRI_ALZHEIMER_DATA"

for folder in os.listdir(root):
    path = os.path.join(root, folder)
    if os.path.isdir(path):
        print(folder, len(os.listdir(path)))

print("\nAbsolute Path:")
print(os.path.abspath(root))
all_labels = []

for images, labels in val_ds:
    all_labels.extend(labels.numpy())

print("Unique:", np.unique(all_labels))
print("Counts:", np.bincount(all_labels))
# تست دقت واقعی مدل
loss, acc = model.evaluate(val_ds, verbose=1)

print("\nEvaluate Accuracy =", acc)
print("Evaluate Loss =", loss)
for i, (images, labels) in enumerate(val_ds.take(5)):
    print("Batch", i)
    print(np.unique(labels.numpy()))
class_names = val_ds.class_names

# جمع‌آوری برچسب‌های واقعی
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("First 20 True Labels:")
print(y_true[:20])

print("First 20 Predicted Labels:")
print(y_pred[:20])
print("Class Names:", class_names)
print("y_true shape:", y_true.shape)
print("Unique y_true:", np.unique(y_true))
print("y_pred shape:", y_pred.shape)
print("Unique y_pred:", np.unique(y_pred))
print("Class Names:", class_names)
print("Counts:", np.bincount(y_true))# گزارش طبقه‌بندی
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0,1,2,3],
        target_names=class_names,
        zero_division=0
    )
)

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

# ذخیره تصویر
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("\nConfusion Matrix ذخیره شد ✅")