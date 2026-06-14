import mt_scan
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

csv_file = "/workspace/data/dataset-synthetic-1/spherical_5_2.csv"

points = np.loadtxt(csv_file, delimiter=",", skiprows=1)
points = points[:, :2].astype(np.float32)

n_points=points.shape[0]

t_start = time.perf_counter()

labels=mt_scan.compute_labels(points, 512, 25)

t_mtscan = time.perf_counter()


print(f"{' -- MTSCAN':<70} | {t_mtscan - t_start:.4f}s")

plt.figure()

noise = labels == -1

plt.scatter(points[~noise, 0], points[~noise, 1],
            c=labels[~noise], s=30)
plt.scatter(points[noise, 0], points[noise, 1],
            c='gray', s=30, alpha=0.7)

plt.savefig("/workspace/tmp/test_plot.png", dpi=150, bbox_inches='tight')