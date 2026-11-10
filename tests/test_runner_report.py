from __future__ import annotations

import json

import numpy as np
import pytest

from zdssl.config import deep_merge
from zdssl.dataset import load_dataset
from zdssl.figures import fig2_budget_trajectory, fig3_operating
from zdssl.report import (budget_trajectory, flatten, load_runs, profile_bot_flows, select,
                          table5a, write_report)
from zdssl.runner import build_matrix, completed_keys, run_stage, scores_path
from zdssl.splits import TEST


def test_matrix_sizes_for_the_paper_config():
    from zdssl.config import load_config
    from conftest import CONFIG

    cfg = load_config(CONFIG)
    assert len(build_matrix(cfg, "pilot")) == 6
    assert len(build_matrix(cfg, "p1")) == 60
    assert len(build_matrix(cfg, "p1")) + len(build_matrix(cfg, "p2")) == 240
    assert len(build_matrix(cfg, "ablation")) == 30 and len(build_matrix(cfg, "condA")) == 10
    keys = [s.key for s in build_matrix(cfg, "all")]
    assert len(keys) == len(set(keys)) == 280
    with pytest.raises(ValueError):
        build_matrix(cfg, "everything")


@pytest.fixture(scope="module")
def finished(pipeline):
    cfg = pipeline["cfg"]
    results = {stage: run_stage(cfg, stage, device="cpu") for stage in ("pilot", "p1", "ablation", "condA")}
    return {**pipeline, "results": results}


def test_run_stage_completes_and_resumes(finished):
    cfg, paths, results = finished["cfg"], finished["paths"], finished["results"]
    assert all(not r["failed"] for r in results.values()), paths["errors"].read_text() if paths["errors"].is_file() else ""
    assert len(results["pilot"]["finished"]) == 6
    assert results["p1"]["skipped"] == 6 and len(results["p1"]["finished"]) == 6  # pilot seed reused
    again = run_stage(cfg, "p1", device="cpu")
    assert again["finished"] == [] and again["skipped"] == 12
    assert len(completed_keys(paths["runs"])) == 12 + 6 + 2
    checkpoints = {p.name for p in paths["ckpt_dir"].iterdir()}
    assert {"F_condB_seed42.pt", "C_condB_seed42.pt", "SCARF_condB_seed42.pt", "A_condB_seed43.pt"} <= checkpoints


def test_torn_last_line_is_rerun_not_fatal(finished, tmp_path):
    runs = tmp_path / "runs.jsonl"
    lines = finished["paths"]["runs"].read_text(encoding="utf-8").splitlines()
    runs.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2], encoding="utf-8")
    assert completed_keys(runs) == {json.loads(lines[0])["key"]}
    assert len(load_runs(runs)) == 1


def test_records_are_complete_and_traceable(finished):
    runs = load_runs(finished["paths"]["runs"])
    by_key = {r["key"]: r for r in runs}
    e06 = by_key["S4_E06|F|B|0.1|42"]
    assert set(e06["zero_day"]) == {"msp", "decoupled", "centroid_benign", "maha_benign", "maha_class", "knn_benign", "raw_knn_benign", "raw_maha_benign"}
    assert e06["closed_world"]["accuracy"] > 0.2
    assert e06["params"]["deployed"] == e06["params"]["pretrain_encoder"] + e06["params"]["probe"]
    assert by_key["DeepSVDD|-|B|0.1|42"]["closed_world"] is None
    assert by_key["RF|-|A|0.1|42"]["n_features"] == by_key["RF|-|B|0.1|42"]["n_features"] + 2
    for record in runs:
        for zd in record["zero_day"].values():
            assert 0 <= zd["auroc"] <= 1
            point = zd["operating_points"]["benign"]
            assert point["tnr"] == pytest.approx(1 - point["fpr"])
            assert point["tp"] + point["fn"] == 400 and point["fp"] + point["tn"] == 400
        assert record["config_hash"] and record["timing"]["adapt_seconds"] >= 0
    saved = np.load(scores_path(finished["paths"]["scores_dir"], e06["key"]))
    assert saved["decoupled_test"].shape == (800,)
    ds = load_dataset(finished["cfg"])
    from zdssl.metrics import ranking_metrics
    assert ranking_metrics(saved["decoupled_test"], ds.test_is_zero_day)["auroc"] == pytest.approx(
        e06["zero_day"]["decoupled"]["auroc"], abs=1e-4)


def test_report_tables_and_figures(finished, tmp_path):
    cfg, paths = finished["cfg"], finished["paths"]
    flat = flatten(load_runs(paths["runs"]))
    tables = write_report(flat, cfg, tmp_path)
    t3 = tables["table3_ranking_closed_world"].set_index("row")
    assert (t3["n_seeds"] == 2).all()
    assert np.isnan(t3.loc["Deep SVDD", "cw_macro_f1_mean"])
    expected = select(flat, "S4_E06", "decoupled", 0.1)["auroc"]
    assert t3.loc["S4_E06 (Decoupled)", "auroc_mean"] == pytest.approx(expected.mean())
    assert t3.loc["S4_E06 (Decoupled)", "auroc_std"] == pytest.approx(expected.std(ddof=1))
    stats = table5a(flat, 0.1)
    assert len(stats) == 6 and stats["p_bh"].between(0, 1).all() and (stats["p_bh"] >= stats["p_raw"]).all()
    assert len(tables["table5b_ablation"]) == 4 and (tables["table5b_ablation"]["n_seeds"] == 2).all()
    assert len(tables["table4b_seed_case_study"]) == 5
    assert len(tables["s4_e06_all_scorers"]) == 8
    assert set(tables["condition_a_shortcut"]["n_seeds"]) == {2}
    assert (tmp_path / "table3_ranking_closed_world.tex").read_text().count(r"\pm") >= 12

    assert fig2_budget_trajectory(budget_trajectory(flat), tmp_path / "fig2.pdf").stat().st_size > 0
    e06 = next(r for r in load_runs(paths["runs"]) if r["key"] == "S4_E06|F|B|0.1|42")
    scores = np.load(scores_path(paths["scores_dir"], e06["key"]))["decoupled_test"]
    ds = load_dataset(cfg)
    taus = {k: v["threshold"] for k, v in e06["zero_day"]["decoupled"]["operating_points"].items()}
    assert fig3_operating(scores, ds.test_is_zero_day, taus, tmp_path / "fig3.pdf").stat().st_size > 0

    test_frame = finished["frame"].loc[finished["frame"]["split"] == TEST]
    profile = profile_bot_flows(test_frame, ds.test_is_zero_day, scores, taus["benign"], "Flow Duration", 1e6)
    assert profile["flows"].sum() == 400
    assert profile["detected"].sum() == e06["zero_day"]["decoupled"]["operating_points"]["benign"]["tp"]


def test_failed_runs_are_recorded_not_swallowed(pipeline, tmp_path):
    cfg = deep_merge(pipeline["cfg"], {"paths": {"drive_root": str(tmp_path / "empty")}})
    result = run_stage(cfg, "pilot", device="cpu")  # no splits under this root
    assert len(result["failed"]) == 6 and result["finished"] == []
    errors = (tmp_path / "empty" / "results" / "errors.jsonl").read_text().splitlines()
    assert len(errors) == 6 and "splits not built" in json.loads(errors[0])["error"]


def test_tuning_never_touches_test_and_is_resumable(pipeline, tmp_path):
    import shutil

    from zdssl.scoring import LATENT_SCORERS
    from zdssl.tune import dev_split, run_tuning, tune_path

    root = tmp_path / "drive"
    shutil.copytree(pipeline["paths"]["drive_root"] / "data", root / "data")
    cfg = deep_merge(pipeline["cfg"], {"paths": {"drive_root": str(root)}})
    ds = load_dataset(cfg)
    held = ds.classes.index("SSH-Bruteforce")
    split = dev_split(ds, held)
    assert held not in split["y_train"]                      # the stand-in class is unseen in training
    assert split["is_novel"].sum() == (ds.y_train == held).sum() + (ds.y_tune == held).sum()
    assert len(split["x_dev"]) < len(ds.x_tune) + len(ds.x_train)
    with pytest.raises(ValueError):
        dev_split(ds, ds.benign_index)

    candidates = {"base": {}, "no_novelty": {"pretrain": {"variants": {"F": [0.5, 0.5, 0.0]}}}}
    summary = run_tuning(cfg, ["SSH-Bruteforce"], [42], candidates, device="cpu")
    assert set(summary.index) == {"base", "no_novelty"}
    assert set(LATENT_SCORERS) | {"msp", "raw_knn_benign", "raw_maha_benign"} == set(summary.columns)
    assert summary.to_numpy().min() >= 0 and summary.to_numpy().max() <= 1
    lines = tune_path(cfg).read_text().splitlines()
    run_tuning(cfg, ["SSH-Bruteforce"], [42], candidates, device="cpu")
    assert tune_path(cfg).read_text().splitlines() == lines   # nothing re-run
    with pytest.raises(ValueError):
        run_tuning(cfg, ["Bot"], [42], candidates, device="cpu")
