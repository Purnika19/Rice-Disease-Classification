import tensorflow as tf
import numpy as np
import cv2
import os

def load_mobilenet_model():
    return tf.keras.models.load_model('models/mobilenetv2.h5')

def get_class_names():
    data_dir = 'data'
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def run_test(image_path):
    img_size = (224, 224)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(image_rgb, img_size) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    model = load_mobilenet_model()
    class_names = get_class_names()
    predictions = model.predict(img_input)
    
    idx = np.argmax(predictions[0])
    return class_names[idx], 100 * predictions[0][idx]

if __name__ == "__main__":
    res1, conf1 = run_test('tests/test_image_1.png')
    res2, conf2 = run_test('tests/test_image_2.png')
    
    print(f"TEST_1_RESULT: {res1} ({conf1:.2f}%)")
    print(f"TEST_2_RESULT: {res2} ({conf2:.2f}%)")
