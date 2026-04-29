import tensorflow as tf
from data_utils import get_data_generators
from sklearn.metrics import classification_report
import pandas as pd
import os
import numpy as np

def evaluate_all():
    DATA_DIR = 'data'
    BATCH_SIZE = 8
    
    _, val_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE)
    class_labels = list(val_gen.class_indices.keys())
    
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    
    all_reports = {}
    
    for model_file in model_files:
        model_name = model_file.replace('.h5', '')
        print(f"Evaluating {model_name}...")
        model_path = os.path.join(models_dir, model_file)
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Predict
        val_gen.reset()
        predictions = model.predict(val_gen)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_gen.classes
        
        # Report
        report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        all_reports[model_name] = report
        
    return all_reports

if __name__ == "__main__":
    reports = evaluate_all()
    for name, report in reports.items():
        print(f"\n--- {name} ---")
        df = pd.DataFrame(report).transpose()
        print(df)
