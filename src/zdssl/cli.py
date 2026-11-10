"""Command-line entry point: ``python -m zdssl.cli {data,run,report} --config configs/paper1.yaml``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Mapping

from .config import config_hash, load_config, resolve_paths, write_text_atomic

log = logging.getLogger("zdssl")

MANIFEST_KEYS = (
    "split_sizes", "n_features_B", "dropped_constant_columns", "quarantine", "sample_class_counts",
    "exact_duplicates_removed", "test_exact_duplicates_removed", "test_rows_removed_seen_in_historical",
    "validation_rows_sharing_a_train_feature_profile", "test_class_counts", "leakage_checks",
)


def cmd_data(cfg: Mapping[str, Any], skip_download: bool) -> None:
    from .data import download_raw, ingest_all, verify_counts
    from .splits import build_splits, save_splits

    paths = resolve_paths(cfg)
    if not skip_download:
        download_raw(cfg)
    day_stats = ingest_all(cfg)
    summary = verify_counts(day_stats, cfg)
    write_text_atomic(paths["ingest_stats"], json.dumps({"summary": summary, "days": day_stats}, indent=2))
    log.info("verified: %d data lines, %d repeated headers, %d flows", summary["raw_rows"],
             summary["header_rows"], summary["clean_rows"])
    log.info("label counts: %s", json.dumps(summary["label_counts"]))
    frame, manifest = build_splits(cfg, day_stats)
    save_splits(frame, manifest, paths["splits"], paths["manifest"])
    for key in MANIFEST_KEYS:
        log.info("%s: %s", key, json.dumps(manifest[key], default=str))
    log.info("DATA DONE: %s", paths["splits"])


def cmd_run(cfg: Mapping[str, Any], stages: list[str]) -> None:
    from .runner import run_stage

    for stage in stages:
        result = run_stage(cfg, stage)
        log.info("STAGE %s DONE: total=%d skipped=%d finished=%d failed=%s", stage, result["total"],
                 result["skipped"], len(result["finished"]), result["failed"])


def cmd_report(cfg: Mapping[str, Any]) -> None:
    from .figures import fig2_budget_trajectory
    from .report import flatten, load_runs, write_report

    paths = resolve_paths(cfg)
    runs = load_runs(paths["runs"], config_hash(cfg))
    tables = write_report(flatten(runs), cfg, paths["report_dir"])
    fig2_budget_trajectory(tables["fig2_budget_trajectory"], paths["report_dir"] / "fig2_budget_trajectory.pdf")
    for name, table in tables.items():
        log.info("## %s\n%s", name, table.round(4).to_string(index=False))
    log.info("REPORT DONE: %d runs -> %s", len(runs), paths["report_dir"])


def cmd_tune(cfg: Mapping[str, Any], held_out: list[str], seeds: list[int], names: list[str] | None) -> None:
    import pandas as pd

    from .tune import CANDIDATES, per_class_table, run_tuning, tune_path

    chosen = {n: CANDIDATES[n] for n in names} if names else None
    summary = run_tuning(cfg, held_out, seeds, chosen)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    log.info("mean dev AUROC (rows: candidate, columns: scorer)\n%s", summary.round(4).to_string())
    log.info("by held-out class\n%s", per_class_table(tune_path(cfg)).round(4).to_string())
    log.info("TUNE DONE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zdssl")
    parser.add_argument("command", choices=("data", "run", "report", "tune"))
    parser.add_argument("--held-out", nargs="+", help="classes used as stand-in zero-days (tune)")
    parser.add_argument("--tune-seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--candidates", nargs="+", help="subset of candidate names (tune)")
    parser.add_argument("stages", nargs="*", help="stages for the run command, e.g. pilot p1")
    parser.add_argument("--config", default="configs/paper1.yaml")
    parser.add_argument("--drive-root", help="override paths.drive_root")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
    overrides = {"paths": {"drive_root": args.drive_root}} if args.drive_root else None
    cfg = load_config(args.config, overrides)
    if args.command == "data":
        cmd_data(cfg, args.skip_download)
    elif args.command == "run":
        if not args.stages:
            parser.error("run needs at least one stage")
        cmd_run(cfg, args.stages)
    elif args.command == "tune":
        if not args.held_out:
            parser.error("tune needs --held-out CLASS [CLASS ...]")
        cmd_tune(cfg, args.held_out, args.tune_seeds, args.candidates)
    else:
        cmd_report(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
