import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--expfolder", type=str)
args = parser.parse_args()

JSON_PATH   = os.path.join(args.expfolder, "experiment.json")
OUTPUT_TEX  = os.path.join(args.expfolder, "table_ari.tex")

with open(JSON_PATH) as f:
    experiment = json.load(f)
lines = []
lines.append(r"\documentclass{article}")
lines.append(r"\usepackage{graphicx}")
lines.append(r"\usepackage{booktabs}")
lines.append(r"\usepackage{longtable}")
lines.append(r"\usepackage[margin=2cm]{geometry}")
lines.append(r"\renewcommand{\arraystretch}{2.5}")
lines.append(r"\begin{document}")

exp_id    = experiment.get("experiment_id", "N/A").replace("_", r"\_")
timestamp = experiment.get("timestamp", "N/A")
params    = experiment.get("parameters", {})
kernel    = params.get("kernel", "N/A")
resolution = params.get("resolution", "N/A")
device    = params.get("device", "N/A")
alpha_range  = params.get("alpha_range", [])
minpts_range = params.get("minpts_range", [])

lines.append(r"\section*{Experiment Parameters}")
lines.append(r"\begin{itemize}")
lines.append(f"  \\item \\textbf{{Experiment ID:}} {exp_id}")
lines.append(f"  \\item \\textbf{{Timestamp:}} {timestamp}")
lines.append(f"  \\item \\textbf{{Kernel:}} {kernel}")
lines.append(f"  \\item \\textbf{{Resolution:}} {resolution}")
lines.append(f"  \\item \\textbf{{Device:}} {device}")
lines.append(f"  \\item \\textbf{{Alpha range:}} {alpha_range}")
lines.append(f"  \\item \\textbf{{Minpts range:}} {minpts_range}")
lines.append(r"\end{itemize}")
lines.append(r"\vspace{1cm}")

lines.append(r"\small")
lines.append(r"\begin{longtable}{|l|c|c|c|c|c|l|}")
lines.append(r"\hline")
lines.append(r"\textbf{Dataset} & \textbf{n\_points} & \textbf{ARI (MT-Scan)} & \textbf{ARI (HDBSCAN)} & \textbf{$\alpha$} & \textbf{minpts} & \textbf{Plot} \\")
lines.append(r"\hline")
lines.append(r"\endfirsthead")
lines.append(r"\hline")
lines.append(r"\textbf{Dataset} & \textbf{n\_points} & \textbf{ARI (MT-Scan)} & \textbf{ARI (HDBSCAN)} & \textbf{$\alpha$} & \textbf{minpts} & \textbf{Plot} \\")
lines.append(r"\hline")
lines.append(r"\endhead")

for result in experiment["results"]:
    dataset     = result["dataset_name"].replace("_", r"\_")
    n_points    = result["n_points"]
    ari_mtscan  = result["mtscan"]["best_ari"]
    ari_hdbscan = result["hdbscan"]["best_ari"]
    best_alpha  = result["mtscan"]["best_alpha_ari"]
    best_minpts = result["hdbscan"]["best_minpts_ari"]
    plot_path   = result["plots"]["ari"]

    row = (
        f"{dataset} & "
        f"{n_points} & "
        f"{ari_mtscan:.4f} & "
        f"{ari_hdbscan:.4f} & "
        f"{best_alpha} & "
        f"{best_minpts} & "
        r"\includegraphics[width=3cm]{" + plot_path + r"} \\ \hline"
    )
    lines.append(row)

lines.append(r"\caption{Best ARI scores per dataset}")
lines.append(r"\label{tab:ari}")
lines.append(r"\end{longtable}")
lines.append(r"\end{document}")

with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(lines))

print(f"Table saved to {OUTPUT_TEX}")