import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from data_utils import get_data_generators
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import os

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 25
DATA_DIR = 'data'

def build_transfer_learning_model(model_name, num_classes):
    print(f"Building {model_name}...")
    if model_name == 'MobileNetV2':
        base_model = applications.MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    elif model_name == 'InceptionV3':
        base_model = applications.InceptionV3(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    elif model_name == 'ResNet50V2':
        base_model = applications.ResNet50V2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    elif model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    elif model_name == 'DenseNet121':
        base_model = applications.DenseNet121(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    elif model_name == 'VGG16':
        base_model = applications.VGG16(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
    else:
        raise ValueError("Unknown model name")
    
    base_model.trainable = False 
    
    model = models.Sequential([
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
    return model, base_model

def train_and_fine_tune(model, base_model, model_name, train_gen, val_gen):
    print(f"\n--- Phase 1: Training top layers for {model_name} ---")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    if base_model is not None:
        print(f"\n--- Phase 2: Fine-tuning {model_name} ---")
        base_model.trainable = True
        # Unfreeze high-level layers
        fine_tune_at = len(base_model.layers) // 4 * 3
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Merge histories
        for key in history.history:
            history.history[key].extend(history_fine.history[key])

    model_save_path = f"{model_name.lower().replace(' ', '_')}.h5"
    model.save(model_save_path)
    return history

def evaluate_metrics(model, model_name, val_gen):
    val_gen.reset()
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())
    
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Greens', annot_kws={"size": 16})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/cm_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return report

def main():
    if not os.path.exists('results'):
        os.makedirs('results')
        
    train_gen, val_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    class_labels = list(train_gen.class_indices.keys())
    
    models_to_train = ['MobileNetV2', 'InceptionV3', 'ResNet50V2', 'EfficientNetB0', 'DenseNet121', 'VGG16']
    
    all_reports = []
    histories = {}
    
    for name in models_to_train:
        try:
            model, base_model = build_transfer_learning_model(name, num_classes)
            history = train_and_fine_tune(model, base_model, name, train_gen, val_gen)
            histories[name] = history
            
            report = evaluate_metrics(model, name, val_gen)
            
            # Extract Overall Metrics
            row = {
                'Model': name,
                'Accuracy': report['accuracy'],
                'Macro Precision': report['macro avg']['precision'],
                'Macro Recall': report['macro avg']['recall'],
                'Macro F1': report['macro avg']['f1-score'],
                'Weighted F1': report['weighted avg']['f1-score']
            }
            
            # Extract Per-Class Metrics
            for label in class_labels:
                row[f'{label}_Precision'] = report[label]['precision']
                row[f'{label}_Recall'] = report[label]['recall']
                row[f'{label}_F1'] = report[label]['f1-score']
                row[f'{label}_Support'] = report[label]['support']
                
            all_reports.append(row)
            print(f"Finished {name}: Accuracy = {report['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            import traceback
            traceback.print_exc()
    
    df_results = pd.DataFrame(all_reports)
    print("\n--- Detailed Results ---")
    print(df_results[['Model', 'Accuracy', 'Macro F1']])
    df_results.to_csv('results/detailed_model_comparison.csv', index=False)
    
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    for name in histories:
        plt.plot(histories[name].history['val_accuracy'], label=name, linewidth=2)
    plt.title('Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('results/accuracy_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()

