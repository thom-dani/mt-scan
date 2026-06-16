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
import re

'''
python3 -u run_exp_time.py > run.log
'''
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", action="store_true",
                    help="Overwrite an existing experiment with the same ID")
parser.add_argument("--hdbscangpu", action="store_true",
                    help="Use cuML HDSBCAN")
args = parser.parse_args()

if(args.hdbscangpu):
    import cupy as cp
    from cuml.cluster import HDBSCAN as cuHDBSCAN

# ─────────────────────────────────────────────
# EXPERIMENT PARAMETERS — fill these in
# ─────────────────────────────────────────────

EXPERIMENT_ID   = "exp_004"
DATASET_FOLDER  = "/workspace/data/time"
EXPERIMENT_FOLDER = os.path.join(DATASET_FOLDER,"results", EXPERIMENT_ID)
PLOTS_FOLDER    = os.path.join(EXPERIMENT_FOLDER, "plots")

if os.path.exists(EXPERIMENT_FOLDER):
    if args.overwrite:
        import shutil
        shutil.rmtree(EXPERIMENT_FOLDER)
        print(f"Overwriting existing experiment '{EXPERIMENT_ID}'")
    else:
        raise FileExistsError(f"There is already an experiment with id {EXPERIMENT_ID}")

os.makedirs(PLOTS_FOLDER)

KERNEL          = "gaussian"       
RESOLUTION      = 512
MTSCAN_DEVICE          = "gpu"
ALPHA_RANGE     = list(range(10, 151, 10))
HDBSCAN_DEVICE = "cpu"
MINPTS_RANGE    = [5, 20, 50]
LOG_PATH = "./run.log"
JSON_PATH = os.path.join(EXPERIMENT_FOLDER, "experiment.json")


os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)


csv_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]

print("=" * 50)
print(f"  EXPERIMENT {EXPERIMENT_ID}")
print("=" * 50)
print(f"  Dataset folder : {DATASET_FOLDER}")
print(f"  Results folder : {EXPERIMENT_FOLDER}")
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

experiment["parameters"]["dataset_list"] = csv_files

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

run_id=0

for csv_file in csv_files:
    dataset_name = csv_file.replace(".csv", "")
    print(f"\n── {dataset_name} ──")
    
    data      = np.loadtxt(os.path.join(DATASET_FOLDER, csv_file), delimiter=",", skiprows=1)
    points    = data[:, :2].astype(np.float32)        
        
    try:
        mt_scan.compute_labels(points, ALPHA_RANGE[0], RESOLUTION, KERNEL, False)
        for alpha in ALPHA_RANGE:

            print(f"run_id:{run_id}")
            t_start = time.perf_counter()
            labels_pred = mt_scan.compute_labels(points=points, alpha=alpha, target_res=RESOLUTION, kernel=KERNEL, print_logs=True)
            elapsed = time.perf_counter() - t_start


            experiment["results"]["mtscan"].append({
                "run_id":   run_id,
                "dataset":    dataset_name,
                "n_points":   len(points),
                "parameters": {"alpha": alpha, "resolution": RESOLUTION, "kernel": KERNEL},
                "time_s":     elapsed
            })
            run_id+=1
    except Exception as e:
        experiment["results"]["mtscan"].append({
                "dataset":    dataset_name,
                "n_points":   len(points),
                "parameters": {"alpha": alpha, "resolution": RESOLUTION, "kernel": KERNEL},
                "time_s":     float("nan")
            })
        print(f"  [MTSCAN]: ERROR on {dataset_name}: {e} — skipping")
    try:    
        if(args.hdsbcangpu):
                _ = cuHDBSCAN(min_cluster_size=50).fit_predict(cp.array(X_f32[:500]))
        else:
            hdbscan.HDBSCAN(min_cluster_size=MINPTS_RANGE[0]).fit_predict(points)

        for minpts in MINPTS_RANGE:
            t_start = time.perf_counter()
            if(args.hdsbcangpu):
                points_gpu = cp.asarray(points, dtype=cp.float32)
                gpu_model = cuHDBSCAN(min_cluster_size=minpts)
                cp.cuda.Stream.null.synchronize()
                gpu_labels_cu = gpu_model.fit_predict(points_gpu)
                cp.cuda.Stream.null.synchronize()
            else:
                clusterer   = hdbscan.HDBSCAN(min_cluster_size=minpts)
                labels_pred = clusterer.fit_predict(points)
            elapsed = time.perf_counter() - t_start

            experiment["results"]["hdbscan"].append({
                "dataset":    dataset_name,
                "n_points":   len(points),
                "parameters": {"minpts": minpts},
                "time_s":     elapsed
            })

    except Exception as e:
        experiment["results"]["hdbscan"].append({
                "dataset":    dataset_name,
                "n_points":   len(points),
                "parameters": {"minpts": minpts},
                "time_s":     float("nan")
            })
        print(f"  [HDBSCAN]: ERROR on {dataset_name}: {e} — skipping")

    

with open(JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)

with open(LOG_PATH) as f:
    log_raw = f.read()

enrich_experiment_json(JSON_PATH, log_raw, True)

print(f"\nResults saved to {JSON_PATH}")

