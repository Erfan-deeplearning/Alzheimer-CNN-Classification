import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# بارگذاری مدل
model = tf.keras.models.load_model("alzheimer_cnn_model.h5")

# کلاس‌ها
classes = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

# مسیر تصویر
img_path =  input("Enter image path: ")
# خواندن تصویر
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)


print("Min:", img_array.min())
print("Max:", img_array.max())
print("Mean:", img_array.mean())
prediction =  model.predict(img_array)
print("Argmax:", np.argmax(prediction))
print("Probabilities:", prediction[0])

predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence:", round(confidence, 2), "%")
print(prediction)