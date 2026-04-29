# Rice Leaf Disease Classification: Actual Project Results

This report provides the extracted performance metrics for 6 models evaluated on the current project dataset (3 classes: BLB, Brown Spot, Leaf Smut).

## 1. Summary Comparison (Section 5.5)

| Model | Params (M) | Test Accuracy (%) | Macro F1 (%) | Comment |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | ~2.2 | **91.7%** | **91.5%** | **Best Efficiency & Accuracy** |
| **ResNet50V2** | ~23.5 | **91.7%** | **91.7%** | Tied for best accuracy |
| **DenseNet121** | ~7.0 | 83.3% | 82.2% | Good balance of params |
| **InceptionV3** | ~21.8 | 79.2% | 78.7% | Higher recall on BLB |
| **VGG16** | ~14.7 | 75.0% | 75.0% | Classic transfer learning baseline |
| **Custom CNN** | ~2.1 | 75.0% | 77.9% | 4 conv blocks — baseline from scratch |

---

## 2. Best Model Performance (MobileNetV2)

### 5.1 Overall Performance
| Metric | Value (%) |
| :--- | :--- |
| Test Accuracy | 91.7% |
| Macro-avg Precision | 93.3% |
| Macro-avg Recall | 91.7% |
| Macro-avg F1-score | 91.5% |
| Weighted F1-score | 91.5% |

### 5.2 Per-class Metrics (MobileNetV2)
| Class | Precision (%) | Recall (%) | F1-score (%) | Support |
| :--- | :--- | :--- | :--- | :--- |

| **Bacterial Leaf Blight** | 100% | 100% | 100% | 8 |
| **Brown Spot** | 80.0% | 100% | 88.9% | 8 |
| **Leaf Smut** | 100% | 75.0% | 85.7% | 8 |

---

## 3. Second Best Model Performance (ResNet50V2)

### 5.1 Overall Performance
| Metric | Value (%) |
| :--- | :--- |
| Test Accuracy | 91.7% |
| Macro-avg Precision | 92.1% |
| Macro-avg Recall | 91.7% |
| Macro-avg F1-score | 91.7% |
| Weighted F1-score | 91.7% |

### 5.2 Per-class Metrics (ResNet50V2)
| Class | Precision (%) | Recall (%) | F1-score (%) | Support |
| :--- | :--- | :--- | :--- | :--- |

| **Bacterial Leaf Blight** | 88.9% | 100% | 94.1% | 8 |
| **Brown Spot** | 100% | 87.5% | 93.3% | 8 |
| **Leaf Smut** | 87.5% | 87.5% | 87.5% | 8 |

---

## 4. Summary for Documentation
The project currently achieves a peak validation accuracy of **91.7%** using both MobileNetV2 and ResNet50V2. The Custom CNN serves as a strong baseline at 75%. Adding the "Healthy" class and increasing the dataset size (currently 40 images/class) is expected to push these accuracies toward the 97-99% target range observed in the literature.
