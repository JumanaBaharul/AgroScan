import numpy as np


def generate_farm_path(H, W):
    """Simple sinusoidal scouting path across the farm."""
    path = []
    for x in range(0, W, 5):
        y = int(H / 2 + 10 * np.sin(x / 10))
        path.append((x, y))
    return path
