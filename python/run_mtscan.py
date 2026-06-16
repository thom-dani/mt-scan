#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mt_scan

CSV_FILE    = "../data/synthetic-1/spherical_6_2.csv"
OUTPUT_PLOT = "../tmp/labels.png"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    points = np.loadtxt(CSV_FILE, delimiter=",", skiprows = 1)
    points = points[:,:2].astype(np.float32)

    labels = mt_scan.compute_labels(points, args.alpha, 512)



    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(f"mt_scan labels (alpha={args.alpha})")
    plt.colorbar(label="label")
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()