print("شروع پروژه آلزایمر 🚀")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# مسیر دیتاست
DATASET_PATH = "dataset/OASIS_1_MRI_ALZHEIMER_DATA"

# بارگذاری دیتاست
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names
print("کلاس‌ها:", class_names)

# بهینه‌سازی
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# مدل CNN
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(4, activation='softmax')
])

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# خلاصه مدل
model.summary()

# آموزش
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# ذخیره مدل
model.save("alzheimer_cnn_model.h5")
import pickle

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("مدل ذخیره شد ✅")