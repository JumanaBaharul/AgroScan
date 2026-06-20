import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from simulate import simulate_sequence, DISEASES


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
MODEL_PATH            = "crop_disease_convlstm.keras"
NUM_SAMPLES_PER_CLASS = 30
T, H, W, C            = 12, 64, 64, 9
NUM_CLASSES            = 3

LABEL_NAMES = ["Pest Attack", "Overwatering", "Water Stress"]

model = load_model(MODEL_PATH)


# ══════════════════════════════════════════════
# TEST DATASET
# ══════════════════════════════════════════════
X, y_class, y_mask = [], [], []

for disease, label_id in DISEASES.items():
    for _ in range(NUM_SAMPLES_PER_CLASS):
        base = np.random.rand(T, H, W, C)
        seq, masks = simulate_sequence(
            base,
            disease=disease,
            max_severity=np.random.uniform(0.3, 0.6)
        )
        noise = np.random.normal(0, 0.02, seq.shape)
        seq   = np.clip(seq + noise, 0, 1)

        X.append(seq)
        y_class.append(label_id)
        y_mask.append(masks[-1][..., None])

X       = np.array(X)
y_class = np.array(y_class)
y_mask  = np.array(y_mask)
y_oh    = to_categorical(y_class, num_classes=NUM_CLASSES)


# ══════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════

pred_cls, pred_msk = model.predict(X, verbose=1)
pred_labels        = np.argmax(pred_cls, axis=1)

# ══════════════════════════════════════════════
# CLASSIFICATION METRICS
# ══════════════════════════════════════════════
print("\n===== DISEASE CLASSIFICATION METRICS =====\n")
acc = accuracy_score(y_class, pred_labels)
print(f"Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_class, pred_labels, target_names=LABEL_NAMES))


# ── Confusion Matrix ──────────────────────────
cm = confusion_matrix(y_class, pred_labels)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix – Disease Classification")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(NUM_CLASSES), LABEL_NAMES, rotation=30, ha="right")
plt.yticks(range(NUM_CLASSES), LABEL_NAMES)
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
plt.tight_layout()
plt.show()


# ── ROC – AUC ────────────────────────────────
plt.figure(figsize=(7, 6))
for idx in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_oh[:, idx], pred_cls[:, idx])
    roc_auc     = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{LABEL_NAMES[idx]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Disease Classification")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════
# SEGMENTATION METRICS
# ══════════════════════════════════════════════
def dice(y_true, y_pred, eps=1e-6):
    inter = np.sum(y_true * y_pred)
    return (2 * inter + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)

def iou(y_true, y_pred, eps=1e-6):
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return (inter + eps) / (union + eps)

dices, ious = [], []
for i in range(len(X)):
    gt   = y_mask[i]
    pred = (pred_msk[i] > 0.5).astype(np.float32)
    dices.append(dice(gt, pred))
    ious.append(iou(gt, pred))

print("\n===== DISEASE LOCALIZATION METRICS =====\n")
print(f"Mean Dice Score : {np.mean(dices):.3f}")
print(f"Mean IoU Score  : {np.mean(ious):.3f}")
print("\n✅  Evaluation completed successfully.")
print("Predicted class distribution:", np.bincount(pred_labels))
