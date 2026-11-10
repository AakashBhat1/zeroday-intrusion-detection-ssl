from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from zdssl.config import ConfigError, deep_merge, load_config
from zdssl.data import ROW_ID, DataVerificationError, clean_chunk, parse_day, verify_counts
from zdssl.dataset import build_dataset
from zdssl.features import SPLIT_COL, fit_preprocessor, select_features
from zdssl.splits import (CW_TEST, TEST, TRAIN, TUNE, LeakageError, budget_subset, check_leakage,
                          dedup_exact, feature_hash)
from zdssl.synth import CONSTANT, N_VARYING

from conftest import CONFIG


def test_config_validation(tmp_path):
    assert load_config(CONFIG)["pretrain"]["epochs"] == 10
    with pytest.raises(ConfigError):
        load_config(CONFIG, {"pretrain": {"warmup_epochs": 10}})
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    base = {"a": {"b": 1}}
    assert deep_merge(base, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}}


def test_parse_day():
    assert parse_day("Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv").isoformat() == "2018-02-20"
    with pytest.raises(DataVerificationError):
        parse_day("notes.csv")


def test_clean_chunk_removes_headers_and_infinities(pipeline):
    cfg = pipeline["cfg"]
    chunk = pd.DataFrame({
        "Dst Port": ["80", "Dst Port", "443"], "Protocol": ["6", "Protocol", "6"],
        "Timestamp": ["14/02/2018 08:00:01", "Timestamp", "bad"],
        "Flow Byts/s": ["Infinity", "Flow Byts/s", "2.5"], "Src IP": ["a", "Src IP", "b"],
        "Label": ["Benign", "Label", "Bot"],
    })
    cleaned, n_header = clean_chunk(chunk, cfg)
    assert n_header == 1 and len(cleaned) == 2
    assert "Src IP" not in cleaned.columns
    assert np.isnan(cleaned["Flow Byts/s"].iloc[0]) and cleaned["Flow Byts/s"].iloc[1] == 2.5
    assert cleaned["Timestamp"].iloc[0] == pd.Timestamp("2018-02-14 08:00:01")
    assert pd.isna(cleaned["Timestamp"].iloc[1])


def test_ingest_counts(pipeline):
    summary, counts = pipeline["summary"], pipeline["counts"]
    assert summary["raw_rows"] == counts["raw_rows"]
    assert summary["header_rows"] == 10
    assert summary["clean_rows"] == counts["raw_rows"] - 10
    assert summary["label_counts"]["Bot"] == counts["zero_day_rows"]
    assert len(summary["label_counts"]) == 15


def test_verify_counts_stops_on_wrong_data(pipeline):
    cfg = deep_merge(pipeline["cfg"], {"data": {"expected_zero_day_rows": 1}})
    with pytest.raises(DataVerificationError):
        verify_counts(pipeline["day_stats"], cfg)
    stats = copy.deepcopy(pipeline["day_stats"])
    stats[0]["label_counts"]["Bot"] = 3  # zero-day flow on a historical day
    with pytest.raises(DataVerificationError):
        verify_counts(stats, pipeline["cfg"])


def test_split_invariants(pipeline):
    frame, manifest = pipeline["frame"], pipeline["manifest"]
    assert all(manifest["leakage_checks"].values())
    assert frame[ROW_ID].is_unique
    assert set(frame[SPLIT_COL]) == {TRAIN, TUNE, CW_TEST, TEST}
    assert not ((frame["Label"] == "Bot") & (frame[SPLIT_COL] != TEST)).any()
    test = frame.loc[frame[SPLIT_COL] == TEST, "Label"].value_counts()
    assert test["Bot"] == test["Benign"] == 400
    assert len(manifest["classes"]) == 14 and "Bot" not in manifest["classes"]
    train_ids = set(frame.loc[frame[SPLIT_COL] == TRAIN, ROW_ID])
    val_ids = set(frame.loc[frame[SPLIT_COL].isin([TUNE, CW_TEST]), ROW_ID])
    assert set(manifest["ssl_pool_row_ids"]) == train_ids and not train_ids & val_ids
    sizes = manifest["split_sizes"]
    historical = sizes[TRAIN] + sizes[TUNE] + sizes[CW_TEST]
    assert abs(sizes[TRAIN] / historical - 0.8) < 0.01
    assert abs(sizes[TUNE] - sizes[CW_TEST]) <= 1
    assert manifest["quarantine"]["seconds_from_train"] >= 12 * 3600
    assert sum(manifest["exact_duplicates_removed"].values()) > 0


def test_features_drop_constant_columns(pipeline):
    manifest = pipeline["manifest"]
    assert manifest["n_features_B"] == N_VARYING
    assert set(manifest["dropped_constant_columns"]) == set(CONSTANT)
    assert manifest["features_A"][:2] == ["Dst Port", "Protocol"]
    assert "Dst Port" not in manifest["features_B"]


def test_leakage_check_catches_violations(pipeline):
    frame, manifest, cfg = pipeline["frame"], pipeline["manifest"], pipeline["cfg"]
    leaked = frame.copy()
    bot = leaked.index[leaked["Label"] == "Bot"][0]
    leaked.loc[bot, SPLIT_COL] = TRAIN
    with pytest.raises(LeakageError, match="no_zero_day_outside_test"):
        check_leakage(leaked, manifest, cfg)

    copied = frame.copy()  # a test flow whose features equal a training flow
    source = copied.index[copied[SPLIT_COL] == TRAIN][0]
    target = copied.index[copied[SPLIT_COL] == TEST][0]
    copied.loc[target, manifest["hash_columns"]] = copied.loc[source, manifest["hash_columns"]].to_numpy()
    with pytest.raises(LeakageError, match="no_feature_duplicates_train_vs_test"):
        check_leakage(copied, manifest, cfg)

    pooled = {**manifest, "ssl_pool_row_ids": [*manifest["ssl_pool_row_ids"],
                                                int(frame.loc[frame[SPLIT_COL] == TUNE, ROW_ID].iloc[0])]}
    with pytest.raises(LeakageError, match="ssl_pool"):
        check_leakage(frame, pooled, cfg)


def test_dedup_exact_ignores_row_id():
    frame = pd.DataFrame({ROW_ID: [1, 2, 3], "f": [1.0, 1.0, 2.0], "Label": ["a", "a", "a"]})
    unique, removed = dedup_exact(frame, "Label")
    assert len(unique) == 2 and removed == {"a": 1}
    hashes = feature_hash(frame, ["f"])
    assert hashes[0] == hashes[1] != hashes[2]


def test_budget_subset_is_stratified_and_deterministic():
    labels = np.repeat(np.arange(4), [1000, 100, 10, 2])
    subset = budget_subset(labels, 0.1, seed=7)
    assert np.bincount(labels[subset]).tolist() == [100, 10, 1, 1]
    assert np.array_equal(subset, budget_subset(labels, 0.1, seed=7))
    assert not np.array_equal(subset, budget_subset(labels, 0.1, seed=8))
    assert len(budget_subset(labels, 1.0, seed=7)) == len(labels)
    with pytest.raises(ValueError):
        budget_subset(labels, 0.0, seed=7)


def test_preprocessor_is_fitted_on_train_only():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, np.nan], "b": [5.0, 5.0, 5.0, 5.0, 9.0]})
    other = pd.DataFrame({"a": [1e9, np.nan], "b": [5.0, -1e9]})
    pre = fit_preprocessor(train, ["a", "b"], clip_abs=10.0)
    assert pre.median.tolist() == [1.5, 5.0]
    out = pre.transform(other)
    assert out.dtype == np.float32 and out[0, 0] == 10.0 and out[1, 1] == -10.0
    assert out[1, 0] == 0.0  # NaN imputed with the training median, which is also the centre
    assert np.array_equal(pre.transform(train), fit_preprocessor(train, ["a", "b"], 10.0).transform(train))
    kept, dropped = select_features(pd.DataFrame({"a": [1.0, 2.0], "z": [0.0, 0.0]}), ["a", "z"])
    assert kept == ["a"] and dropped == ["z"]


def test_build_dataset_scaler_ignores_other_splits(pipeline):
    frame, manifest, cfg = pipeline["frame"], pipeline["manifest"], pipeline["cfg"]
    ds = build_dataset(frame, manifest, cfg, "B")
    shifted = frame.copy()
    not_train = shifted[SPLIT_COL] != TRAIN
    shifted.loc[not_train, manifest["features_B"]] = shifted.loc[not_train, manifest["features_B"]] * 3 + 1
    ds_shifted = build_dataset(shifted, manifest, cfg, "B")
    assert np.array_equal(ds.x_train, ds_shifted.x_train)  # train scaling unaffected by other splits
    assert ds.x_train.shape[1] == N_VARYING and ds.n_classes == 14
    assert ds.test_is_zero_day.sum() == 400 and np.abs(ds.x_test).max() <= 10.0
    assert build_dataset(frame, manifest, cfg, "A").x_train.shape[1] == N_VARYING + 2
    with pytest.raises(ValueError):
        build_dataset(frame, manifest, cfg, "C")
