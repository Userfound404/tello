from tensorflow import keras
model= keras.models.load_model("mobilenet_classifier_finetuned.keras")
print("Model loaded Successfully!")