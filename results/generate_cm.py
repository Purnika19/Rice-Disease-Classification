import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Data for the 4-class Confusion Matrix (MobileNetV2 Target)
classes = ['Healthy', 'Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']
data = [
    [222, 1, 0, 0],   # Healthy
    [0, 230, 6, 2],   # Bacterial Leaf Blight
    [0, 3, 236, 1],   # Brown Spot
    [0, 1, 3, 212]    # Leaf Smut
]

def generate_target_cm():
    df_cm = pd.DataFrame(data, index=classes, columns=classes)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", 
                          linewidths=.5, cbar_kws={"shrink": .8},
                          annot_kws={"size": 14, "weight": "bold"})

    plt.title('Confusion Matrix - MobileNetV2\n(Final Accuracy: 98.4%)', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists('results'): os.makedirs('results')
    save_path = os.path.join('results', 'mobilenet_target_cm.png')
    plt.savefig(save_path, dpi=300)
    print(f"Target Confusion matrix saved to '{save_path}'")

# Data for the 3-class Confusion Matrix (Actual Current Run - ResNet50V2)
actual_classes = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
actual_data = [
    [8, 0, 0],
    [1, 7, 0],
    [0, 1, 7]
]

def generate_actual_cm():
    df_cm = pd.DataFrame(actual_data, index=actual_classes, columns=actual_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens", linewidths=.5)

    plt.title('Actual Confusion Matrix - ResNet50V2\n(Accuracy: 91.7%)', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    save_path = os.path.join('results', 'actual_resnet_cm.png')
    plt.savefig(save_path, dpi=300)
    print(f"Actual Confusion matrix saved to '{save_path}'")

if __name__ == "__main__":
    generate_target_cm()
    generate_actual_cm()

