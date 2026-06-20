import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from simulate import simulate_sequence
from optimized_path import compare_algorithms


# ══════════════════════════════════════════════
# 3-CLASS LABEL MAPPING
# ══════════════════════════════════════════════
DISEASE_LABELS = {
    0: "Pest Attack",
    1: "Overwatering",
    2: "Water Stress"
}

model = load_model("crop_disease_convlstm.keras")

T, C           = 12, 9
PLOT_H, PLOT_W = 64, 64
GRID_Y, GRID_X = 4, 4
FARM_H         = GRID_Y * PLOT_H
FARM_W         = GRID_X * PLOT_W


# ══════════════════════════════════════════════
# FARM MAPS
# ══════════════════════════════════════════════
farm_gt   = np.zeros((FARM_H, FARM_W))
farm_prob = np.zeros((FARM_H, FARM_W))


# ══════════════════════════════════════════════
# PLOT-WISE INFERENCE
# ══════════════════════════════════════════════
for gy in range(GRID_Y):
    for gx in range(GRID_X):
        disease = np.random.choice(["pest", "overwater", "water_stress"])
        base    = np.random.rand(T, PLOT_H, PLOT_W, C)

        seq, gt_masks = simulate_sequence(
            base,
            disease=disease,
            max_severity=np.random.uniform(0.3, 0.5)
        )

        noise = np.random.normal(0, 0.02, seq.shape)
        seq   = np.clip(seq + noise, 0, 1)

        sample   = np.expand_dims(seq, axis=0)
        cls, msk = model.predict(sample, verbose=0)

        pred_id   = int(np.argmax(cls))
        conf      = float(np.max(cls)) * 100
        prob_mask = msk[0, ..., 0]

        y0, y1 = gy * PLOT_H, (gy+1) * PLOT_H
        x0, x1 = gx * PLOT_W, (gx+1) * PLOT_W

        farm_gt[y0:y1, x0:x1]   = gt_masks[-1]
        farm_prob[y0:y1, x0:x1] = prob_mask

        print(f"  Plot ({gy},{gx}): simulated={disease:12s}  "
              f"predicted={DISEASE_LABELS[pred_id]:12s}  conf={conf:.1f}%")


# ══════════════════════════════════════════════
# SPRAY ZONES
# ══════════════════════════════════════════════
norm_farm = (farm_prob - farm_prob.min()) / (farm_prob.max() - farm_prob.min() + 1e-6)
threshold = np.percentile(norm_farm, 90)
farm_spray = norm_farm > threshold

print("\nFarm probability  min:", round(farm_prob.min(), 4),
      " max:", round(farm_prob.max(), 4))
print("Total spray pixels  :", farm_spray.sum())


# ══════════════════════════════════════════════
# VISUALISATION – IMAGE 1
# ══════════════════════════════════════════════
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.title("Entire Farm – Ground Truth")
plt.imshow(farm_gt, cmap="Reds")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Entire Farm – Predicted Probability")
plt.imshow(farm_prob, cmap="Reds")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Entire Farm – Spray Zones")
plt.imshow(farm_prob, cmap="Reds")
plt.imshow(farm_spray.astype(np.uint8), cmap="gray", alpha=0.45)
plt.axis("off")

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════
# VISUALISATION – IMAGE 2 + METRICS
# ══════════════════════════════════════════════
np.save("entire_farm_ground_truth.npy",  farm_gt)
np.save("entire_farm_probability.npy",   farm_prob)
np.save("entire_farm_spray_mask.npy",    farm_spray.astype(np.uint8))

df, paths = compare_algorithms(farm_prob, farm_spray, start=(0, 0))

print("\n✅  Entire farm inference + algorithm comparison completed.")
