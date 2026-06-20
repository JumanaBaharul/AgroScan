import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from simulate import simulate_sequence
from paths import generate_farm_path


# ══════════════════════════════════════════════
# 3-CLASS LABEL MAPPING  (matches training)
# ══════════════════════════════════════════════
DISEASE_LABELS = {
    0: "Pest Attack",
    1: "Overwatering",
    2: "Water Stress"
}


# ══════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════
model = load_model("crop_disease_convlstm.keras")

T, H, W, C = 12, 64, 64, 9

# Change "pest" → "overwater" / "water_stress" to test other classes
SIMULATED_DISEASE = "pest"

base = np.random.rand(T, H, W, C)
seq, gt_masks = simulate_sequence(
    base,
    disease=SIMULATED_DISEASE,
    max_severity=0.6
)

noise = np.random.normal(0, 0.02, seq.shape)
seq   = np.clip(seq + noise, 0, 1)
sample = np.expand_dims(seq, axis=0)


# ══════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════
cls, mask = model.predict(sample)

disease_id   = int(np.argmax(cls))
disease_name = DISEASE_LABELS[disease_id]
confidence   = float(np.max(cls)) * 100
prob_mask    = mask[0, ..., 0]


# ══════════════════════════════════════════════
# SPRAY ZONE
# ══════════════════════════════════════════════
norm_prob  = (prob_mask - prob_mask.min()) / (prob_mask.max() - prob_mask.min() + 1e-6)
threshold  = np.percentile(norm_prob, 85)
spray_mask = norm_prob > threshold


# ══════════════════════════════════════════════
# CONSOLE OUTPUT
# ══════════════════════════════════════════════
print(f"\nSimulated Disease : {SIMULATED_DISEASE}")
print(f"Predicted Disease : {disease_name}")
print(f"Confidence        : {confidence:.2f}%")
print(f"Spray Pixels      : {spray_mask.sum()}")

farm_path = generate_farm_path(H, W)
print(f"Farmer Path (first 5 pts): {farm_path[:5]}")


# ══════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Ground Truth Disease Mask")
plt.imshow(gt_masks[-1], cmap="Reds")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Predicted: {disease_name} ({confidence:.1f}%)")
plt.imshow(prob_mask, cmap="Reds")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Spray Zones (Thresholded)")
plt.imshow(prob_mask, cmap="Reds")
plt.imshow(spray_mask.astype(np.uint8), cmap="gray", alpha=0.45)
plt.axis("off")

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════
np.save("predicted_probability_mask.npy", prob_mask)
np.save("spray_mask.npy",                 spray_mask.astype(np.uint8))
np.save("predicted_disease_class.npy",    cls)

print("\n✅  Inference completed and outputs saved.")
