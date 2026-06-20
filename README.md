# AgroScan

**Precision Agriculture Disease Detection & Treatment Route Optimization**

AgroScan is a deep learning-powered web application that detects crop diseases from multispectral satellite imagery, localizes affected areas at pixel level, and generates optimized treatment routes for autonomous spraying systems (drones and ground robots).

---

## Features

- **Temporal Disease Detection** — ConvLSTM2D model analyzes 12-day time-series data to classify three disease types: Pest Attack, Overwatering, and Water Stress
- **Pixel-Level Localization** — Multi-task segmentation head produces a disease probability map for precise spray targeting
- **5-Algorithm Route Optimization** — Benchmarks Direct A\*, MST + A\*, TSP (nearest-neighbor), Dijkstra, and Simulated Annealing; selects the most efficient path automatically
- **GPS Waypoint Export** — Converts pixel coordinates to real-world GPS coordinates and exports a drone-ready CSV
- **Interactive Dashboard** — Streamlit UI with live heatmaps, algorithm comparison metrics, and downloadable reports

---

## Project Structure

```
AgroScan/
├── app.py                      # Streamlit web dashboard (main entry point)
├── model.py                    # ConvLSTM2D model architecture
├── train.py                    # Model training pipeline
├── simulate.py                 # Synthetic Sentinel-2 disease data generator
├── optimized_path.py           # Path optimization algorithms (A*, MST, TSP, Dijkstra, SA)
├── inference.py                # Single-plot inference script
├── full_farm_inference.py      # Full 16-plot farm inference script
├── evaluate.py                 # Model evaluation (accuracy, ROC, Dice, IoU)
├── paths.py                    # Simple scouting path utility
├── crop_disease_convlstm.keras # Pre-trained model weights
├── requirements.txt            # Python dependencies
└── *.npy                       # Cached inference outputs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15+ / Keras (ConvLSTM2D) |
| Web UI | Streamlit 1.32+ |
| Visualization | Plotly 5.18+ |
| Image Processing | OpenCV, SciPy |
| Data | NumPy, Pandas, Scikit-learn |
| Language | Python 3.11+ |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/JumanaBaharul/AgroScan.git
cd AgroScan

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

1. Open the dashboard at `http://localhost:8501`
2. Use the **sidebar** to set a random seed, choose a spray threshold percentile (80–98%), and toggle visualizations
3. Click **Run Farm Analysis** — the app simulates a 4×4 plot farm, runs inference on each plot, and displays results
4. Review the **disease heatmap**, **stress classification grid**, and **algorithm comparison table**
5. Download the **GPS waypoint CSV** for direct upload to your drone or spraying robot

---

## Model Architecture

```
Input: (batch, 12 timesteps, 64×64 px, 9 spectral bands)
        │
        ├─ ConvLSTM2D(64, 3×3, return_sequences=True) → BatchNorm
        ├─ ConvLSTM2D(32, 3×3, return_sequences=True) → BatchNorm
        └─ ConvLSTM2D(16, 3×3, return_sequences=False) → BatchNorm
                         │
          ┌──────────────┴──────────────┐
          │                             │
  GlobalAvgPool → Dropout          Conv2D(1, 1×1)
  → Dense(3, softmax)               sigmoid
  [Disease Class]                  [Spray Mask]
```

**Training details:**
- 360 synthetic samples (120 per class), 80/20 train/val split
- Loss: 70% categorical cross-entropy (classification) + 30% binary cross-entropy (segmentation)
- Optimizer: Adam with learning rate reduction on plateau
- Early stopping: patience = 8 on `val_disease_accuracy`

---

## Disease Classes

| Class | Label | Spectral Signature |
|---|---|---|
| 0 | Pest Attack | NIR↓, RED↑, GREEN↓, RedEdge↑ |
| 1 | Overwatering | SWIR↑, BLUE↑, NIR↓ (excess moisture) |
| 2 | Water Stress | SWIR↓, NIR↓, RED↑ (dry soil) |

Spectral signatures are modeled after Sentinel-2 band responses (9 bands: Coastal, Blue, Green, Red, VNIR, RedEdge, NIR, SWIR, Extra).

---

## Path Optimization Algorithms

| Algorithm | Strategy |
|---|---|
| Direct A\* | Visit disease centroids in detection order |
| MST + A\* | Prim's MST on centroids, then A\* between nodes |
| TSP | Nearest-neighbor greedy heuristic |
| Dijkstra | Classic shortest path between centroids |
| Simulated Annealing | Metaheuristic order optimization, then A\* |

All algorithms share the same A\* cost map (1 − disease probability), so high-disease areas have lower traversal cost. The best algorithm is selected automatically by **efficiency = disease gain / distance**.

---

## GPS Output

The farm bounding box is mapped to a region near **Thanjavur, Tamil Nadu, India**:

| Parameter | Value |
|---|---|
| Latitude range | 10.7500 – 10.7564 |
| Longitude range | 79.1300 – 79.1364 |

Each spray waypoint in the exported CSV contains `latitude`, `longitude`, `pixel_x`, `pixel_y`, and `disease_probability`.

---

## Training From Scratch

```bash
# Generate synthetic data and retrain the model
python train.py

# Evaluate on the held-out validation set
python evaluate.py

# Run inference on a single plot
python inference.py

# Run inference on the full 16-plot farm
python full_farm_inference.py
```

---

## Architecture Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Streamlit UI)                       │
│   seed / spray threshold / visualization toggles → Run button    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FARM SIMULATION LAYER                        │
│  simulate.py                                                      │
│  • 4×4 grid → 16 plots (each 64×64 px)                          │
│  • Assign random disease class per plot                          │
│  • Generate 12-timestep Sentinel-2-like spectral data            │
│  • Apply disease severity ramp (30% → 100%) + sensor noise      │
└───────────────────────────┬─────────────────────────────────────┘
                            │  (batch, 12, 64, 64, 9) per plot
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONVLSTM2D INFERENCE ENGINE                    │
│  model.py  |  crop_disease_convlstm.keras                        │
│                                                                   │
│  ConvLSTM2D(64) → ConvLSTM2D(32) → ConvLSTM2D(16)              │
│        │                                                          │
│   ┌────┴─────┐                                                    │
│   │          │                                                    │
│ Classification      Segmentation                                  │
│ Dense(3, softmax)  Conv2D(1, sigmoid)                            │
│ [disease type]     [disease mask]                                 │
└──────┬────────────────────┬────────────────────────────────────-─┘
       │                    │
       ▼                    ▼
┌──────────────┐   ┌────────────────────────────────────────────┐
│ Disease Type │   │           PROBABILITY MAP                   │
│ per plot     │   │  16 masks stitched into 256×256 farm map   │
│ (0/1/2)      │   │  Normalized → Percentile threshold         │
└──────────────┘   │  → Binary spray mask                       │
                   └──────────────────┬─────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PATH OPTIMIZATION LAYER                       │
│  optimized_path.py                                               │
│                                                                   │
│  Extract disease centroids                                       │
│  Run 5 algorithms in parallel:                                   │
│   Direct A*  │  MST+A*  │  TSP  │  Dijkstra  │  Sim. Annealing │
│                                                                   │
│  Rank by Efficiency = Gain / Distance                            │
│  → Best algorithm selected automatically                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │  pixel waypoints
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GPS CONVERSION LAYER                          │
│  Pixel (x, y) → (latitude, longitude)                           │
│  Farm bounding box: Thanjavur, Tamil Nadu region                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴───────────────┐
              ▼                             ▼
┌─────────────────────────┐   ┌────────────────────────────────┐
│   DASHBOARD OUTPUTS     │   │    AUTONOMOUS SYSTEM OUTPUT    │
│  • Disease heatmap      │   │   CSV with GPS waypoints       │
│  • Spray route overlay  │   │   → Drone / spraying robot     │
│  • Stress class grid    │   │     mission planning           │
│  • Algorithm metrics    │   └────────────────────────────────┘
│  • Per-plot GPS report  │
└─────────────────────────┘
```

---

## License

This project is for academic and research purposes.

---

## Author

**Jumana Baharul** — [deepikajumana@gmail.com](mailto:deepikajumana@gmail.com)
