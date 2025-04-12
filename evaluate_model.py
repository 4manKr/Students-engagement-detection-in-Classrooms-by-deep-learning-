import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Load Model ===
model = tf.keras.models.load_model("emotion_model2.h5")

# === Reload Data (same preprocessing as training) ===
BASE_PATH = "C:/Users/common-research/Desktop/dataset"  # Change as needed
FER_Train = "C:/Users/common-research/Desktop/dataset/train"
FER_Test = "C:/Users/common-research/Desktop/dataset/test"

train_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(FER_Train,
                                                 batch_size=512,
                                                 target_size=(48, 48),
                                                 color_mode='grayscale',
                                                 class_mode='categorical',
                                                 shuffle=False)

# === Predict ===
y_pred_probs = model.predict(training_set)
y_pred = np.argmax(y_pred_probs, axis=1)

# === Ground truth ===
y_true = training_set.classes

# === Class labels ===
class_labels = {v: k for k, v in training_set.class_indices.items()}

# === Confusion Matrix & Report ===
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels.values()))

# === ROC Curve Note ===
# roc_curve doesn't support multi-class directly without binarizing
# Skipping ROC plot unless labels are binarized (can be done if needed)

# === Custom Metrics ===
def cal_ma(cm):
    total = 0
    for i in range(len(cm)):
        TP = cm[i][i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = cm.sum() - (TP + FP + FN)
        P = FP + TP
        N = TN + FN
        cal_tp = TP / P if P != 0 else 0
        cal_tn = TN / N if N != 0 else 0
        total += (cal_tp + cal_tn)
    mA = total / (2 * len(cm))
    return mA

def cal_Acc(cm):
    total = 0
    for i in range(len(cm)):
        TP = cm[i][i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = cm.sum() - (TP + FP + FN)
        total += (TP + TN) / (TP + TN + FP + FN)
    return total / len(cm)

def cal_Prec(cm):
    total = 0
    for i in range(len(cm)):
        TP = cm[i][i]
        FP = sum(cm[:, i]) - TP
        if TP + FP == 0:
            continue
        total += TP / (TP + FP)
    return total / len(cm)

def cal_Rec(cm):
    total = 0
    for i in range(len(cm)):
        TP = cm[i][i]
        FN = sum(cm[i, :]) - TP
        if TP + FN == 0:
            continue
        total += TP / (TP + FN)
    return total / len(cm)

def cal_F1(cm):
    prec = cal_Prec(cm)
    rec = cal_Rec(cm)
    return 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

# === Print Custom Metrics ===
print(f"\nMean Accuracy: {cal_ma(cm):.4f}")
print(f"Accuracy: {cal_Acc(cm):.4f}")
print(f"Precision: {cal_Prec(cm):.4f}")
print(f"Recall: {cal_Rec(cm):.4f}")
print(f"F1 Score: {cal_F1(cm):.4f}")
