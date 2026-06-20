'''
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Conv2D, Dropout
)
from tensorflow.keras.models import Model


def build_model(T, H, W, C, num_classes=3):
    """
    ConvLSTM2D backbone with two heads:
      • disease  – softmax over num_classes (3: pest / overwater / water_stress)
      • mask     – sigmoid segmentation map

    num_classes is 3 (Healthy excluded — never trained on it).
    """
    inputs = Input(shape=(T, H, W, C))

    x = ConvLSTM2D(
        64, 3, padding="same",
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )(inputs)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(
        32, 3, padding="same",
        return_sequences=False,
        dropout=0.2
    )(x)
    x = BatchNormalization()(x)

    shared = x

    # ── Disease classification head ──────────────────────────
    gap = GlobalAveragePooling2D()(shared)
    gap = Dropout(0.3)(gap)
    disease = Dense(num_classes, activation="softmax", name="disease")(gap)

    # ── Segmentation mask head ───────────────────────────────
    mask = Conv2D(1, 1, activation="sigmoid", name="mask")(shared)

    model = Model(inputs, [disease, mask])

    model.compile(
        optimizer="adam",
        loss={
            "disease": "categorical_crossentropy",
            "mask":    "binary_crossentropy"
        },
        loss_weights={
            "disease": 0.4,
            "mask":    0.6
        },
        metrics={
            "disease": "accuracy"
        }
    )

    return model
'''
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Conv2D, Dropout
)
from tensorflow.keras.models import Model


def build_model(T, H, W, C, num_classes=3):
    """
    3-layer ConvLSTM backbone.

    Key fix vs previous version:
      - Layer 2 now uses return_sequences=True  (was False)
        so layer 3 sees the full temporal context before collapsing.
      - Added a 3rd ConvLSTM that collapses the sequence.
      - Dropout before the classification Dense layer.

    Outputs
    -------
    disease : softmax over num_classes  (3 = pest / overwater / water_stress)
    mask    : sigmoid segmentation map  (H x W x 1)
    """
    inputs = Input(shape=(T, H, W, C))

    # ── Layer 1: extract spatial-temporal features ───────────
    x = ConvLSTM2D(
        64, 3, padding="same",
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2
    )(inputs)
    x = BatchNormalization()(x)

    # ── Layer 2: deepen representation, keep sequence ────────
    x = ConvLSTM2D(
        32, 3, padding="same",
        return_sequences=True,      # ← was False; now passes full seq to L3
        dropout=0.2,
        recurrent_dropout=0.1
    )(x)
    x = BatchNormalization()(x)

    # ── Layer 3: collapse sequence into single feature map ───
    x = ConvLSTM2D(
        16, 3, padding="same",
        return_sequences=False,     # ← final collapse here
        dropout=0.1
    )(x)
    x = BatchNormalization()(x)

    shared = x   # shape: (batch, H, W, 16)

    # ── Classification head ───────────────────────────────────
    gap     = GlobalAveragePooling2D()(shared)
    gap     = Dropout(0.3)(gap)
    disease = Dense(num_classes, activation="softmax", name="disease")(gap)

    # ── Segmentation head ─────────────────────────────────────
    mask = Conv2D(1, 1, activation="sigmoid", name="mask")(shared)

    model = Model(inputs, [disease, mask])

    model.compile(
        optimizer="adam",
        loss={
            "disease": "categorical_crossentropy",
            "mask":    "binary_crossentropy"
        },
        # loss_weights={
        #     "disease": 0.5,     # bumped from 0.4 → 0.5 for better cls
        #     "mask":    0.5
        # },
        loss_weights={
            "disease": 0.7,   # was 0.5
            "mask":    0.3    # was 0.5
        },
        metrics={
            "disease": "accuracy"
        }
    )

    return model
