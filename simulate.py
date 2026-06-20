'''import numpy as np
import cv2
import random

# 3-class mapping (no Healthy — never trained on it)
DISEASES = {
    "pest":         0,
    "overwater":    1,
    "water_stress": 2
}


def random_blob_mask(shape, coverage=0.30):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = mask.shape
    num_blobs = max(1, int(coverage * 10))
    for _ in range(num_blobs):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(6, 18)
        cv2.circle(mask, (x, y), r, 1, -1)
    return mask.astype(bool)


def pest_attack(img, severity):
    # equal coverage as others → 0.30
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 7] *= (1 - severity)          # NIR  ↓
    img[mask, 3] *= (1 + severity)           # RED  ↑
    return img, mask


def overwatering(img, severity):
    # was 0.40 — equalised to 0.30 to remove bias
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 8] *= (1 + severity)           # SWIR ↑
    img[mask, 7] *= (1 - severity / 2)       # NIR  ↓
    return img, mask


def water_stress(img, severity):
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 7] *= (1 - severity)           # NIR  ↓
    img[mask, 3] *= (1 + severity / 2)       # RED  ↑
    return img, mask


def simulate_sequence(images, disease, max_severity=0.6):
    """
    Ramp severity from 30 % → 100 % of max_severity so early
    timesteps are NOT near-zero (which caused the model to always
    pick the noisiest pattern it had memorised = Overwatering).
    """
    out, masks = [], []
    T = len(images)

    for t, img in enumerate(images):
        # FIX: start at 30 % severity, not 0 %
        severity = max_severity * (0.3 + 0.7 * t / max(T - 1, 1))
        img = img.copy()

        if disease == "pest":
            img, mask = pest_attack(img, severity)
        elif disease == "overwater":
            img, mask = overwatering(img, severity)
        elif disease == "water_stress":
            img, mask = water_stress(img, severity)
        else:
            mask = np.zeros(img.shape[:2], dtype=bool)

        out.append(np.clip(img, 0, 1))
        masks.append(mask)

    return np.array(out), np.array(masks)
'''
import numpy as np
import cv2
import random

# 3-class mapping (no Healthy)
DISEASES = {
    "pest":         0,
    "overwater":    1,
    "water_stress": 2
}


def random_blob_mask(shape, coverage=0.30):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    h, w = mask.shape
    num_blobs = max(1, int(coverage * 10))
    for _ in range(num_blobs):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(6, 18)
        cv2.circle(mask, (x, y), r, 1, -1)
    return mask.astype(bool)


'''
def pest_attack(img, severity):
    """
    Sentinel-2 bands (0-indexed):
      0=Coastal  1=Blue  2=Green  3=Red  4=VNIR
      5=RedEdge1 6=RedEdge2 7=NIR  8=SWIR

    Pest signature: NIR↓  RED↑  GREEN↓  RedEdge1↑
    """
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 7] *= (1 - severity)           # NIR       ↓  plant damage
    img[mask, 3] *= (1 + severity)           # RED       ↑  chlorophyll loss
    img[mask, 2] *= (1 - severity * 0.5)     # GREEN     ↓  leaf damage
    img[mask, 5] *= (1 + severity * 0.3)     # RedEdge1  ↑  stress shift
    return img, mask


def overwatering(img, severity):
    """
    Overwatering signature: SWIR↑  NIR↓  BLUE↑  Coastal↑
    Key: SWIR goes UP  (excess moisture)  ←→ opposite of water_stress
    """
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 8] *= (1 + severity)           # SWIR      ↑  soil moisture HIGH
    img[mask, 7] *= (1 - severity * 0.5)     # NIR       ↓
    img[mask, 2] *= (1 + severity * 0.4)     # BLUE      ↑  water reflectance
    img[mask, 0] *= (1 + severity * 0.3)     # Coastal   ↑  water signature
    return img, mask


def water_stress(img, severity):
    """
    Water stress signature: NIR↓  SWIR↓  RED↑(mild)  VNIR↓
    Key: SWIR goes DOWN  (dry soil)  ←→ opposite of overwatering
    """
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 7] *= (1 - severity * 0.6)     # NIR       ↓  (softer than pest)
    img[mask, 8] *= (1 - severity * 0.5)     # SWIR      ↓  DRY — key discriminator
    img[mask, 3] *= (1 + severity * 0.3)     # RED       ↑  mild
    img[mask, 4] *= (1 - severity * 0.4)     # VNIR      ↓
    return img, mask
'''
def pest_attack(img, severity):
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 7] *= (1 - severity * 0.8)     # NIR    ↓ strong
    img[mask, 3] *= (1 + severity * 0.8)     # RED    ↑ strong
    img[mask, 2] *= (1 - severity * 0.6)     # GREEN  ↓
    img[mask, 5] *= (1 + severity * 0.5)     # RedEdge↑
    img[mask, 8] *= (1 + severity * 0.1)     # SWIR   slightly up (pest ≠ dry)
    return img, mask

def overwatering(img, severity):
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 8] *= (1 + severity * 1.0)     # SWIR   ↑↑ KEY: moisture high
    img[mask, 7] *= (1 - severity * 0.3)     # NIR    ↓ mild
    img[mask, 2] *= (1 + severity * 0.6)     # BLUE   ↑
    img[mask, 0] *= (1 + severity * 0.5)     # Coastal↑
    img[mask, 3] *= (1 - severity * 0.2)     # RED    ↓ mild (opposite of pest)
    return img, mask

def water_stress(img, severity):
    mask = random_blob_mask(img.shape, coverage=0.30)
    img[mask, 8] *= (1 - severity * 0.9)     # SWIR   ↓↓ KEY: dry soil
    img[mask, 7] *= (1 - severity * 0.5)     # NIR    ↓ moderate
    img[mask, 3] *= (1 + severity * 0.2)     # RED    ↑ mild
    img[mask, 4] *= (1 - severity * 0.6)     # VNIR   ↓
    img[mask, 2] *= (1 - severity * 0.3)     # GREEN  ↓ (opposite of overwater)
    return img, mask

def simulate_sequence(images, disease, max_severity=0.6):
    """
    Ramp severity 30% → 100% of max_severity.
    Avoids near-zero early frames that caused class-collapse bias.
    """
    out, masks = [], []
    T = len(images)

    for t, img in enumerate(images):
        severity = max_severity * (0.3 + 0.7 * t / max(T - 1, 1))
        img = img.copy()

        if disease == "pest":
            img, mask = pest_attack(img, severity)
        elif disease == "overwater":
            img, mask = overwatering(img, severity)
        elif disease == "water_stress":
            img, mask = water_stress(img, severity)
        else:
            mask = np.zeros(img.shape[:2], dtype=bool)

        out.append(np.clip(img, 0, 1))
        masks.append(mask)

    return np.array(out), np.array(masks)
