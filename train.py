'''
import numpy as np
import random
from simulate import simulate_sequence, DISEASES
from model import build_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ══════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════
T, H, W, C   = 12, 64, 64, 9
NUM_CLASSES   = 3          # pest / overwater / water_stress  (no Healthy)
SAMPLES_PER_DISEASE = 80   # was 40 — more data = better generalisation
NUM_BASE_SCENES     = 40   # was 20


# ══════════════════════════════════════════════
# BASE SCENES
# ══════════════════════════════════════════════
bases = [np.random.rand(T, H, W, C) for _ in range(NUM_BASE_SCENES)]

X, y_class, y_mask = [], [], []


# ══════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════
for disease, label in DISEASES.items():
    for _ in range(SAMPLES_PER_DISEASE):
        base = random.choice(bases)

        seq, masks = simulate_sequence(
            base,
            disease=disease,
            max_severity=random.uniform(0.3, 0.7)
        )

        noise = np.random.normal(0, 0.02, seq.shape)
        seq   = np.clip(seq + noise, 0, 1)

        X.append(seq)
        y_class.append(label)
        y_mask.append(masks[-1][..., None])

X       = np.array(X)
y_class = to_categorical(y_class, num_classes=NUM_CLASSES)
y_mask  = np.array(y_mask)

print(f"Dataset: {X.shape[0]} samples  |  classes: {NUM_CLASSES}")


# ══════════════════════════════════════════════
# TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════
(X_train, X_val,
 yc_train, yc_val,
 ym_train, ym_val) = train_test_split(
    X, y_class, y_mask,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ══════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════
model = build_model(T, H, W, C, num_classes=NUM_CLASSES)
model.summary()


# ══════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ══════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════
model.fit(
    X_train,
    {"disease": yc_train, "mask": ym_train},
    validation_data=(
        X_val,
        {"disease": yc_val, "mask": ym_val}
    ),
    epochs=50,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)


# ══════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════
model.save("crop_disease_convlstm.keras")
print("\n✅  Model saved as crop_disease_convlstm.keras")

'''

import numpy as np
import random
from collections import Counter

from simulate import simulate_sequence, DISEASES
from model import build_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ══════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════
T, H, W, C          = 12, 64, 64, 9
NUM_CLASSES          = 3          # pest=0 / overwater=1 / water_stress=2
SAMPLES_PER_DISEASE  = 120        # was 80  → more data per class
NUM_BASE_SCENES      = 60         # was 40


# ══════════════════════════════════════════════════════
# BASE SCENES  (random Sentinel-2-like images)
# ══════════════════════════════════════════════════════
bases = [np.random.rand(T, H, W, C) for _ in range(NUM_BASE_SCENES)]

X, y_class, y_mask = [], [], []


# ══════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION  + AUGMENTATION
# ══════════════════════════════════════════════════════
for disease, label_id in DISEASES.items():
    for _ in range(SAMPLES_PER_DISEASE):
        base = random.choice(bases)

        seq, masks = simulate_sequence(
            base,
            disease=disease,
            max_severity=random.uniform(0.3, 0.7)
        )

        # Sensor noise
        noise = np.random.normal(0, 0.02, seq.shape)
        seq   = np.clip(seq + noise, 0, 1)

        # ── Spatial augmentation ──────────────────────
        if np.random.rand() > 0.5:
            seq   = seq[:, ::-1, :, :]       # vertical flip
            masks = masks[:, ::-1, :]
        if np.random.rand() > 0.5:
            seq   = seq[:, :, ::-1, :]       # horizontal flip
            masks = masks[:, :, ::-1]

        X.append(seq)
        y_class.append(label_id)
        y_mask.append(masks[-1][..., None])

X       = np.array(X)
y_class = np.array(y_class)
y_mask  = np.array(y_mask)

# ── Balance check (should be 120 / 120 / 120) ────────
counts = Counter(y_class)
print("\nClass distribution before one-hot encoding:")
for k, v in sorted(counts.items()):
    print(f"  class {k}: {v} samples")
assert len(set(counts.values())) == 1, \
    "WARNING: Classes are imbalanced — check DISEASES dict!"

y_class_oh = to_categorical(y_class, num_classes=NUM_CLASSES)

print(f"\nTotal samples : {len(X)}")
print(f"X shape       : {X.shape}")


# ══════════════════════════════════════════════════════
# TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════
(X_train, X_val,
 yc_train, yc_val,
 ym_train, ym_val) = train_test_split(
    X, y_class_oh, y_mask,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y_class        # keep class balance in both splits
)


# ══════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════
model = build_model(T, H, W, C, num_classes=NUM_CLASSES)
model.summary()


# ══════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════
callbacks = [
    EarlyStopping(
        monitor="val_disease_accuracy",
        patience=8,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# ══════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════
# class_weight ensures equal gradient contribution even if
# a batch happens to be skewed
class_weight = {0: 1.0, 1: 1.0, 2: 1.0}

# model.fit(
#     X_train,
#     {"disease": yc_train, "mask": ym_train},
#     validation_data=(
#         X_val,
#         {"disease": yc_val, "mask": ym_val}
#     ),
#     epochs=60,
#     batch_size=8,
#     callbacks=callbacks,
#     class_weight={"disease": class_weight},
#     verbose=1
# )
model.fit(
    X_train,
    {"disease": yc_train, "mask": ym_train},
    validation_data=(
        X_val,
        {"disease": yc_val, "mask": ym_val}
    ),
    epochs=60,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════
model.save("crop_disease_convlstm.keras")
print("\n✅  Model saved as crop_disease_convlstm.keras")
