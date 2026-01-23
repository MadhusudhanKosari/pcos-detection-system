import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/image_cnn_classifier.h5"

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_image_probability(image_path):
    img = preprocess_image(image_path)
    prob = model.predict(img)[0][0]
    return float(prob)
