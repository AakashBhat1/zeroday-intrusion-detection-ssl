"""Generate the three thin Colab notebooks from the cell lists below.

Run: ``uv run python tools/build_notebooks.py``
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''\
# Setup: mount Drive and import the project code.
# Put the project folder (src/, configs/) at MyDrive/paper1/code, or set REPO_URL to a git remote.
from google.colab import drive
drive.mount("/content/drive")

import logging, subprocess, sys
REPO_URL = ""  # optional
CODE = "/content/drive/MyDrive/paper1/code"
if REPO_URL:
    CODE = "/content/zdssl"
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE], check=False)
    subprocess.run(["git", "-C", CODE, "pull"], check=True)
sys.path.insert(0, f"{CODE}/src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

from zdssl.config import config_hash, load_config, resolve_paths
cfg = load_config(f"{CODE}/configs/paper1.yaml")
paths = resolve_paths(cfg)
print("code:", CODE, "| results:", paths["runs"])
'''

DATA = [
    ("md", "# 00 Data: download, verify, build splits\n\nRuns once. Safe to re-run after a disconnect: "
           "the download and each cleaned day are skipped when already present."),
    ("code", SETUP),
    ("code", '''\
# 1. Download the official CSVs (about 7 GB) to the Colab disk.
!pip -q install awscli
from zdssl.data import download_raw
download_raw(cfg)
!ls -lh "{cfg['paths']['raw_dir']}"
'''),
    ("code", '''\
# 2. Clean each day to parquet and verify the dataset. Stops with an error if the counts are wrong.
import json
from zdssl.data import ingest_all, verify_counts
day_stats = ingest_all(cfg)
summary = verify_counts(day_stats, cfg)
paths["ingest_stats"].parent.mkdir(parents=True, exist_ok=True)
paths["ingest_stats"].write_text(json.dumps({"summary": summary, "days": day_stats}, indent=2))
print("data lines incl. repeated headers:", summary["raw_rows"])
print("repeated header lines removed:   ", summary["header_rows"])
print("flows after cleaning:            ", summary["clean_rows"])
summary["label_counts"]
'''),
    ("code", '''\
# 3. Build the splits. Raises LeakageError if any invariant fails; nothing is saved in that case.
from zdssl.splits import build_splits, save_splits
frame, manifest = build_splits(cfg, day_stats)
save_splits(frame, manifest, paths["splits"], paths["manifest"])
print("saved:", paths["splits"])
'''),
    ("code", '''\
# 4. Numbers to send back / put in the paper (all also stored in manifest.json on Drive).
for key in ("split_sizes", "n_features_B", "dropped_constant_columns", "quarantine",
            "exact_duplicates_removed", "test_exact_duplicates_removed",
            "test_rows_removed_seen_in_historical",
            "validation_rows_sharing_a_train_feature_profile", "sample_class_counts",
            "leakage_checks"):
    print(f"\\n{key}:\\n  {manifest[key]}")
if manifest["quarantine"]["max_hour_of_day_seen"] <= 12:
    print("\\nNOTE: timestamps use a 12-hour clock without AM/PM; the quarantine gap is a lower bound.")
print("\\ncalibration benign flows:", manifest["split_class_counts"]["tune"][cfg["data"]["benign_label"]])
'''),
]

RUNS = [
    ("md", "# 01 Runs: pilot, then the resumable matrix\n\nEvery finished run appends one line to "
           "`results/runs.jsonl` on Drive. Re-running a cell after a disconnect skips what is done."),
    ("code", SETUP),
    ("code", '''\
import torch
from zdssl.runner import build_matrix, run_stage
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU, slow)")
'''),
    ("code", '''\
# Pilot: every model once (seed 42, 10% budget). Measures wall time and extrapolates.
import pandas as pd
from zdssl.report import flatten, load_runs
result = run_stage(cfg, "pilot")
print("failed:", result["failed"])
pilot_keys = {s.key for s in build_matrix(cfg, "pilot")}
pilot = [r for r in load_runs(paths["runs"], config_hash(cfg)) if r["key"] in pilot_keys]
timing = pd.DataFrame([{"model": r["model"], "wall_s": r["timing"]["wall_seconds"],
                        "pretrain_s": r["timing"].get("pretrain_seconds", 0.0)} for r in pilot])
per_seed = timing["wall_s"].sum()
print(timing.to_string(index=False))
print(f"\\nestimate for the 240-run matrix: about {per_seed * 40 / 60:.0f} min "
      "(upper bound: pretraining is paid once per seed, not once per budget)")
'''),
    ("code", '''\
# Priority 1: 10% budget, 10 seeds, all models (Tables 3, 4, 5A).
result = run_stage(cfg, "p1"); print({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
'''),
    ("code", '''\
# Priority 2: the other budgets (Fig. 2).
result = run_stage(cfg, "p2"); print({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
'''),
    ("code", '''\
# Priority 3 and 4: objective ablation (Table 5B) and Condition A (shortcut check).
for stage in ("ablation", "condA"):
    result = run_stage(cfg, stage); print({k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
'''),
    ("code", '''\
# Status: what is done, and any failures (full tracebacks are in results/errors.jsonl).
from zdssl.runner import completed_keys
done = completed_keys(paths["runs"], config_hash(cfg))
for stage in ("p1", "p2", "ablation", "condA"):
    specs = build_matrix(cfg, stage)
    print(f"{stage:9s} {sum(s.key in done for s in specs):3d} / {len(specs)}")
if paths["errors"].is_file():
    print("\\nerrors logged:"); print(paths["errors"].read_text()[-3000:])
'''),
]

REPORT = [
    ("md", "# 02 Report: tables and figures from runs.jsonl\n\nEvery number for the paper is produced "
           "here and written to `MyDrive/paper1/report/` (CSV at full precision, plus LaTeX)."),
    ("code", SETUP),
    ("code", '''\
import json
import numpy as np, pandas as pd
from IPython.display import display
from zdssl.report import flatten, load_runs, write_report, budget_trajectory, profile_bot_flows
from zdssl.runner import build_matrix, completed_keys, scores_path
pd.set_option("display.max_columns", 40, "display.width", 250, "display.float_format", "{:.4f}".format)

runs = load_runs(paths["runs"], config_hash(cfg))  # runs made with any other config are left out
flat = flatten(runs)
done = {r["key"] for r in runs}
for stage in ("p1", "p2", "ablation", "condA"):
    specs = build_matrix(cfg, stage)
    print(f"{stage:9s} {sum(s.key in done for s in specs):3d} / {len(specs)} runs present")
print("devices:", sorted(flat["device"].unique()), "| config hashes:", sorted({r["config_hash"] for r in runs}))
'''),
    ("code", '''\
# All tables. Check n_seeds in each: a value below 10 means that row is not complete yet.
tables = write_report(flat, cfg, paths["report_dir"])
for name, table in tables.items():
    print("\\n##", name); display(table)
'''),
    ("code", '''\
# Figures 2 and 3 (Fig. 3 uses the pilot seed, 10% budget, decoupled score).
from zdssl.dataset import load_dataset
from zdssl.figures import fig2_budget_trajectory, fig3_operating
fig2_budget_trajectory(tables["fig2_budget_trajectory"], paths["report_dir"] / "fig2_budget_trajectory.pdf")

seed, budget = cfg["matrix"]["pilot_seed"], cfg["matrix"]["primary_budget"]
case = next(r for r in runs if r["key"] == f"S4_E06|F|B|{budget:g}|{seed}")
scores = np.load(scores_path(paths["scores_dir"], case["key"]))["decoupled_test"]
ds = load_dataset(cfg)
taus = {k: v["threshold"] for k, v in case["zero_day"]["decoupled"]["operating_points"].items()}
fig3_operating(scores, ds.test_is_zero_day, taus, paths["report_dir"] / "fig3_operating.pdf",
               title_suffix=f" (seed {seed})")
print("AUROC", case["zero_day"]["decoupled"]["auroc"], taus)
'''),
    ("code", '''\
# Zero-day flow profiling: long sessions vs short keepalive flows.
# This is a description of the test flows, not model tuning. Look at the distribution, pick the cut,
# and state the rule in the paper.
from zdssl.splits import load_splits
frame, manifest = load_splits(paths["splits"], paths["manifest"])
test_frame = frame.loc[frame["split"] == "test"]
bot = test_frame.loc[test_frame[cfg["data"]["label_column"]] == cfg["data"]["zero_day_label"]]
display(bot[["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"]].describe(percentiles=[.5, .9, .95, .99]))

PROFILE_COLUMN, PROFILE_CUT = "Flow Duration", 1_000_000   # microseconds; edit after looking above
profile = profile_bot_flows(test_frame, ds.test_is_zero_day, scores, taus["benign"], PROFILE_COLUMN, PROFILE_CUT)
profile.to_csv(paths["report_dir"] / "bot_flow_profile.csv", index=False)
profile
'''),
    ("code", '''\
# Protocol numbers for Section 4.1, Fig. 1 and the parameter counts.
ingest = json.loads(paths["ingest_stats"].read_text())["summary"]
case_params = case["params"]
numbers = {
    "flows_after_cleaning": ingest["clean_rows"], "repeated_header_lines": ingest["header_rows"],
    "zero_day_flows_in_dataset": ingest["label_counts"][cfg["data"]["zero_day_label"]],
    "split_sizes": manifest["split_sizes"], "n_features_B": manifest["n_features_B"],
    "calibration_benign_flows": manifest["split_class_counts"]["tune"][cfg["data"]["benign_label"]],
    "quarantine": manifest["quarantine"],
    "validation_rows_sharing_a_train_feature_profile": manifest["validation_rows_sharing_a_train_feature_profile"],
    "budget_train_sizes": {f"{b:g}": int(flat.loc[np.isclose(flat["budget"], b) & (flat["model"] == "RF")
                                                  & (flat["condition"] == "B"), "n_train"].median())
                           for b in sorted(flat["budget"].unique(), reverse=True)},
    "parameters": case_params,
}
(paths["report_dir"] / "protocol_numbers.json").write_text(json.dumps(numbers, indent=2))
numbers
'''),
]


def notebook(cells: list[tuple[str, str]]) -> dict:
    def cell(kind: str, source: str) -> dict:
        base = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            base.update(execution_count=None, outputs=[])
        return base

    return {
        "cells": [cell(kind, source) for kind, source in cells],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in (("00_data", DATA), ("01_runs", RUNS), ("02_report", REPORT)):
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        compile_errors = []
        for kind, source in cells:
            if kind == "code":
                body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("!"))
                try:
                    compile(body, name, "exec")
                except SyntaxError as error:
                    compile_errors.append(str(error))
        if compile_errors:
            raise SystemExit(f"{name}: {compile_errors}")
        print(f"wrote {path}")  # noqa: T201 - command-line tool output


if __name__ == "__main__":
    main()
