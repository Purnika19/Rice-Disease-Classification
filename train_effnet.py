import tensorflow as tf
from tensorflow.keras import layers, optimizers, applications
from data_utils import get_data_generators
from train_models import evaluate_metrics
import pandas as pd
import os

def main():
    DATA_DIR = 'data'
    BATCH_SIZE = 8
    
    train_gen, val_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    
    name = 'EfficientNetB0'
    print(f"Training {name}...")
    
    # Use weights=None to avoid the bug, but then we must train the whole model
    base_model = applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None)
    base_model.trainable = True 
    
    model = tf.keras.models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    # Train for a bit to get some results
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        verbose=1
    )
    
    report = evaluate_metrics(model, name, val_gen)
    
    row = {
        'Model': name,
        'Accuracy': report['accuracy'],
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall']
    }
    
    df = pd.read_csv('results/model_comparison.csv')
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv('results/model_comparison.csv', index=False)
    
    print("Done training EfficientNetB0.")

if __name__ == "__main__":
    main()
