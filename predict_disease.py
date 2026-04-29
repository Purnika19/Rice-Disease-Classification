import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os

def load_mobilenet_model():
    model_path = 'models/mobilenetv2.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find the MobileNetV2 model at {model_path}. Please ensure training is complete.")
        
    print(f"Loading MobileNetV2 model...")
    return tf.keras.models.load_model(model_path)


def get_class_names():
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return ['Bacterial leaf blight', 'Brown spot', 'Leaf smut'] # Default fallback
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return classes

def predict():

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    

    print("Please select an image file...")
    file_path = filedialog.askopenfilename(
        title="Select Rice Leaf Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        print("No file selected. Exiting.")
        return

    
    img_size = (224, 224)
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not read the image.")
        return
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
   
    img_input = cv2.resize(image_rgb, img_size)
    img_input = img_input / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    
    try:
        model = load_mobilenet_model()
        class_names = get_class_names()
        
        predictions = model.predict(img_input)
        
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = 100 * predictions[0][predicted_idx]

        # 5. Display Result
        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        color = 'green' if confidence > 80 else 'orange' if confidence > 50 else 'red'
        plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", 
                  fontsize=16, color=color, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        print(f"\n--- Prediction Result ---")
        print(f"Disease: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"--------------------------")
        plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict()

