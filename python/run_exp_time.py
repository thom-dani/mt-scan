import os
import json
import numpy as np
import hdbscan
import mt_scan
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
import argparse
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action="store_true",
                    help="Overwrite an existing experiment with the same ID")
args = parser.parse_args()

# ─────────────────────────────────────────────
# EXPERIMENT PARAMETERS — fill these in
# ─────────────────────────────────────────────

EXPERIMENT_ID   = "exp_time"
DATASET_FOLDER  = "/workspace/data/data-time"
RESULTS_FOLDER  = os.path.join(DATASET_FOLDER, "results")
PLOTS_FOLDER    = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID, "plots")



experiment_folder = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID)

if os.path.exists(experiment_folder):
    if args.overwrite:
        import shutil
        shutil.rmtree(experiment_folder)
        print(f"Overwriting existing experiment '{EXPERIMENT_ID}'")
    else:
        raise FileExistsError(f"There is already an experiment with id {EXPERIMENT_ID}")

os.makedirs(PLOTS_FOLDER)


KERNEL          = "gaussian"       
RESOLUTION      = 512
DEVICE          = "cpu"

ALPHA_RANGE     = list(range(25, 500, 25))
MINPTS_RANGE    = [5, 10, 15, 20, 25, 30, 40, 50]

csv_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

print("=" * 50)
print(f"  EXPERIMENT {EXPERIMENT_ID}")
print("=" * 50)
print(f"  Dataset folder : {DATASET_FOLDER}")
print(f"  Results folder : {experiment_folder}")
print(f"  Device         : {DEVICE}")
print(f"  Kernel         : {KERNEL}")
print(f"  Resolution     : {RESOLUTION}")
print(f"  Alpha range    : {list(ALPHA_RANGE)}")
print(f"  MinPts range   : {MINPTS_RANGE}")
print(f"  Datasets       : {csv_files}")
print("=" * 50)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
os.makedirs(PLOTS_FOLDER, exist_ok=True)

experiment = {
    "experiment_id": EXPERIMENT_ID,
    "timestamp": datetime.now().isoformat(),
    "parameters": {
        "kernel":       KERNEL,
        "resolution":   RESOLUTION,
        "device":       DEVICE,
        "alpha_range":  ALPHA_RANGE,
        "minpts_range": MINPTS_RANGE,
        "dataset_list": []
    },
    "results": {"mtscan": [], 
                "hdbscan": []}
}


import re
import json

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
 
 
def parse_timer_logs(log_text: str):
    chunks = re.split(r"run_id\s*:\s*(\d+)", log_text)
 
    parsed: dict[int, dict] = {}
 
    it = iter(chunks)
    next(it)
 
    for run_id_str, block in zip(it, it):
        run_id = int(run_id_str)
        timers = {}
        for match in re.finditer(r"TIMER\s+(\w+)\s+([\d.]+)", block):
            name, value = match.group(1), float(match.group(2))
            timers[name] = value
        if timers:
            parsed[run_id] = timers
 
    return parsed
 
 
def enrich_experiment_json(json_path: str, log_text: str, save: bool = True) -> dict:
    
    with open(json_path) as f:
        experiment = json.load(f)
 
    timer_map = parse_timer_logs(log_text)
 
    matched = 0
    missing_run_ids = []
 
    for entry in experiment["results"]["mtscan"]:
        run_id = entry.get("run_id")
        if run_id in timer_map:
            entry["timers"] = timer_map[run_id]
            matched += 1
        else:
            missing_run_ids.append(run_id)
 
    print(f"Matched {matched}/{len(experiment['results']['mtscan'])} mtscan entries.")
    if missing_run_ids:
        print(f"  No timer data found for run_ids: {missing_run_ids}")
 
    unmatched_log_ids = sorted(set(timer_map) - {e.get("run_id") for e in experiment["results"]["mtscan"]})
    if unmatched_log_ids:
        print(f"  Log contained run_ids not found in JSON: {unmatched_log_ids}")
 
    if save:
        with open(json_path, "w") as f:
            json.dump(experiment, f, indent=2)
        print(f"  Saved enriched experiment to {json_path}")
 
    return experiment

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict


def build_time_matrix(json_path: str) -> dict:
    with open(json_path) as f:
        experiment = json.load(f)

    # collect all unique axis values
    n_points_set = set()
    alpha_set    = set()

    for entry in experiment["results"]["mtscan"]:
        n_points_set.add(entry["n_points"])
        alpha_set.add(entry["parameters"]["alpha"])

    n_points_vals = sorted(n_points_set)
    alpha_vals    = sorted(alpha_set)

    n_pts_idx = {v: i for i, v in enumerate(n_points_vals)}
    alpha_idx = {v: i for i, v in enumerate(alpha_vals)}

    matrix = np.full((len(n_points_vals), len(alpha_vals)), np.nan)

    for entry in experiment["results"]["mtscan"]:
        timers = entry.get("timers")
        if timers is None:
            continue
        i = n_pts_idx[entry["n_points"]]
        j = alpha_idx[entry["parameters"]["alpha"]]
        matrix[i, j] = timers.get("TOTAL", np.nan)

    return {
        "matrix":   matrix,
        "n_points": n_points_vals,
        "alphas":   alpha_vals,
    }



def plot_time_vs_npoints(
    time_data,
    alpha_range,
    title,
    save_path
):
    matrix    = time_data["matrix"]
    n_points  = time_data["n_points"]
    alphas    = time_data["alphas"]

    alphas_to_plot = alpha_range if alpha_range is not None else alphas
    # keep only alphas that actually exist in the data
    alphas_to_plot = [a for a in alphas_to_plot if a in alphas]

    colors = cm.plasma(np.linspace(0.1, 0.9, len(alphas_to_plot)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, alpha in zip(colors, alphas_to_plot):
        j    = alphas.index(alpha)
        col  = matrix[:, j]
        mask = ~np.isnan(col)
        ax.plot(
            np.array(n_points)[mask],
            col[mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"α={alpha}",
        )

    ax.set_xlabel("n_points")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.legend(title="alpha", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    return ax



def plot_time_vs_alpha(
    time_data,
    npoints_range,
    title,
    save_path
):
    matrix   = time_data["matrix"]
    n_points = time_data["n_points"]
    alphas   = time_data["alphas"]

    npts_to_plot = npoints_range if npoints_range is not None else n_points
    npts_to_plot = [n for n in npts_to_plot if n in n_points]

    colors = cm.viridis(np.linspace(0.1, 0.9, len(npts_to_plot)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for color, npts in zip(colors, npts_to_plot):
        i    = n_points.index(npts)
        row  = matrix[i, :]
        mask = ~np.isnan(row)
        ax.plot(
            np.array(alphas)[mask],
            row[mask],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=f"n={npts:,}",
        )

    ax.set_xlabel("alpha")
    ax.set_ylabel("TOTAL time (s)")
    ax.set_title(title)
    ax.legend(title="n_points", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()

    return ax


# ─────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────


experiment["parameters"]["dataset_list"] = csv_files

for csv_file in csv_files:
    dataset_name = csv_file.replace(".csv", "")
    print(f"\n── {dataset_name} ──")
    run_id=0
    try:
        data      = np.loadtxt(os.path.join(DATASET_FOLDER, csv_file), delimiter=",", skiprows=1)
        points    = data[:, :2].astype(np.float32)

        for alpha in ALPHA_RANGE:

            print(f"run_id:{run_id}")
            t_start = time.perf_counter()
            labels_pred = mt_scan.compute_labels(points, alpha, RESOLUTION, KERNEL, True)
            elapsed = time.perf_counter() - t_start


            experiment["results"]["mtscan"].append({
                "run_id":   run_id,
                "dataset":    dataset_name,
                "n_points":   len(points),
                "parameters": {"alpha": alpha, "resolution": RESOLUTION, "kernel": KERNEL},
                "time_s":     elapsed
            })
            run_id+=1

        #for minpts in MINPTS_RANGE:
        #    t_start = time.perf_counter()
        #    clusterer   = hdbscan.HDBSCAN(min_cluster_size=minpts)
        #    labels_pred = clusterer.fit_predict(points)
        #    elapsed = time.perf_counter() - t_start
#
        #    experiment["results"]["hdbscan"].append({
        #        "dataset":    dataset_name,
        #        "n_points":   len(points),
        #        "parameters": {"min_cluster_size": minpts},
        #        "time_s":     elapsed
        #    })

    except Exception as e:
        print(f"  ERROR on {dataset_name}: {e} — skipping")
        continue
JSON_PATH = os.path.join(RESULTS_FOLDER, EXPERIMENT_ID, "experiment.json")
os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
with open(JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

print(f"\nResults saved to {JSON_PATH}")


LOG_PATH = "./run.log"

with open(LOG_PATH) as f:
    log_raw = f.read()

enrich_experiment_json(JSON_PATH, log_raw, True)
time_data = build_time_matrix(JSON_PATH)

#save_path=os.path.join(PLOTS_FOLDER, "time_vs_n_points.png")
save_path=f"{PLOTS_FOLDER}/time_vs_n_points.png"

alpha_range=[25, 50, 75, 100]
title="Runtime vs number of points"

plot_time_vs_npoints(
        time_data,
        alpha_range,
        title,
        save_path,
    )

npoints_range= [100_000, 200_000, 500_000, 1000000]
title="Runtime vs alpha"
#save_path=os.path.join(PLOTS_FOLDER, "time_vs_alpha.png"),
save_path=f"{PLOTS_FOLDER}/time_vs_alpha.png"

plot_time_vs_alpha(
        time_data,
        npoints_range,
        title,
        save_path
    )

