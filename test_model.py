import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_SIZE = 256
CLASS_NAMES = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

def load_and_preprocess_image(image_path):
    # Load the image
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    # Convert the image to a numpy array and normalize
    image = img_to_array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def classify_flower(model_path, image_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the input image
    processed_image = load_and_preprocess_image(image_path)
    
    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    print(f"Predicted Class: {predicted_class} ({confidence:.2f}% confidence)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_model.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    classify_flower(model_path, image_path)
