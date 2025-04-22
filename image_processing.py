import torch
from realesrgan import RealESRGAN
from PIL import Image
import numpy as np
import cv2

# Function to enhance image quality using Real-ESRGAN
def enhance_image_realesrgan(img):
    # Initialize the Real-ESRGAN model (download and use the pretrained model)
    model = RealESRGAN.from_pretrained("RealESRGAN_x4")
    model.eval()

    # Convert BGR to RGB for PIL compatibility
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Enhance the image using Real-ESRGAN
    enhanced_img = model.predict(img_pil)

    # Convert back to BGR for OpenCV
    enhanced_img_bgr = np.array(enhanced_img)[..., ::-1]

    return enhanced_img_bgr

# Function to enhance the image using the given file path
def enhance_image_from_file(file_path):
    # Read the image from the given file path
    img = cv2.imread(file_path)
    
    if img is not None:
        # Enhance the image using Real-ESRGAN
        enhanced_img = enhance_image_realesrgan(img)
        
        # Save the enhanced image to a new file
        enhanced_image_path = "enhanced_image.jpg"  # You can change the name or path here
        cv2.imwrite(enhanced_image_path, enhanced_img)

        # Display the original and enhanced images
        cv2.imshow("Original Image", img)
        cv2.imshow("Enhanced Image", enhanced_img)

        # Wait until a key is pressed to close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Enhanced image saved to: {enhanced_image_path}")
    else:
        print("Failed to load image!")

# Provide the file path of the image to be enhanced
file_path = 'SwinIR/captured_frames/anihant.jpg'  # Replace with your actual file path
enhance_image_from_file(file_path)
