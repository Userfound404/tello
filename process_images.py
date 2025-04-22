import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

# Load the trained model
model = load_model("mobilenet_classifier_finetuned.keras")

# Define threshold for classification
THRESHOLD = 5  # You can adjust this as needed

# Define paths
desktop_path = str(Path.home()) + "\\Desktop\\"
input_folder = desktop_path + "Captured_Frames\\"
good_folder = desktop_path + "Good_Images\\"
bad_folder = desktop_path + "Bad_Images\\"

# Make sure the output folders exist
os.makedirs(good_folder, exist_ok=True)
os.makedirs(bad_folder, exist_ok=True)
# Loop over all the images in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    if os.path.isfile(img_path):
        # Load the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
        img_array = img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image for MobileNetV2

        # Make the prediction
        score = model.predict(img_array)[0][0]

        print(f"Image: {img_name}, Score: {score}")
        # Classify based on threshold
        if score >= THRESHOLD:
            # Move to Good_Images folder
            shutil.move(img_path, os.path.join(good_folder, img_name))
            print(f"Moved {img_name} to Good_Images.")
        else:
            # Move to Bad_Images folder
            shutil.move(img_path, os.path.join(bad_folder, img_name))
            print(f"Moved {img_name} to Bad_Images.")

    print("Image processing complete!")

