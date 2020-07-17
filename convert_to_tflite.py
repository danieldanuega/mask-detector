import tensorflow as tf
from tensorflow.python.keras.models import load_model

print("[INFO] Load keras model")
model = load_model("mask_detector.model")

print("[INFO] Convert keras model to tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("[INFO] Write tflite to disk")
with tf.io.gfile.GFile("mask_detector.tflite", "wb") as f:
    f.write(tflite_model)
