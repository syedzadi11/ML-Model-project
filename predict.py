import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# 1. Load trained model
# =========================
# model = tf.keras.models.load_model("E:/ML_Model/model.keras")
model = tf.keras.models.load_model("model.keras")

# =========================
# 2. Class names 
# =========================
class_names = ["Cat", "Dog"]

# =========================
# 3. Image path 
# =========================
img_path = "test-image/test-1.jpg"

# =========================
# 4. Load & preprocess image
# =========================
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # batch dimension
img_array = img_array / 255.0  # same scaling as training

# =========================
# 5. Prediction
# =========================
prediction = model.predict(img_array)

# =========================
# 6. Result
# =========================
predicted_class = class_names[np.argmax(prediction)]

print("Prediction Probabilities:", prediction)
print("Final Result:", predicted_class)