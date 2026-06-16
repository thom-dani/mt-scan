from sklearn.datasets import (
    make_blobs, make_moons, make_circles, 
    make_classification, make_s_curve, make_swiss_roll
)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FOLDER = Path(f"../data/time/")

def save_dataset(name, X, y):
    n_k_points = X.shape[0]//1000
    data = np.column_stack((X[:, :2], y))
    filename = Path(FOLDER) / f"{name}_{n_k_points}.csv"
    np.savetxt(filename, data, delimiter=",", header="x,y,label", comments="")



n_points_range = range(100_000, 2_000_001, 100_000)

for n in n_points_range:

    save_dataset(
            "blobs_grid",
            *make_blobs(n_samples=n, centers=9, n_features=2, cluster_std=0.4, random_state=42)
        )