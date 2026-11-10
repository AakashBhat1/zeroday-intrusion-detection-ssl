from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wilcoxon

from zdssl.calibrate import oracle_threshold, percentile_threshold
from zdssl.metrics import closed_world_metrics, evaluate_zero_day, operational_metrics, ranking_metrics
from zdssl.scoring import centroid_distance, msp_score
from zdssl.stats import benjamini_hochberg, cliffs_delta, delta_magnitude, wilcoxon_exact


def test_scores():
    assert centroid_distance(np.array([[3.0, 4.0], [0.0, 0.0]]), np.zeros(2)).tolist() == [25.0, 0.0]
    assert np.allclose(msp_score(np.array([[0.7, 0.3], [0.5, 0.5]])), [0.3, 0.5])


def test_percentile_threshold_gives_target_fpr():
    scores = np.arange(1000, dtype=float)
    tau = percentile_threshold(scores, 0.05)
    assert np.mean(scores > tau) == pytest.approx(0.05, abs=0.001)
    with pytest.raises(ValueError):
        percentile_threshold(np.array([]), 0.05)


def test_operational_metrics_known_confusion():
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    m = operational_metrics(scores, labels, 0.5, realistic_prevalence=0.01)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["tnr"] == pytest.approx(1 - m["fpr"]) and m["fnr"] == pytest.approx(1 - m["recall"])
    expected = (2 / 3 * 0.01) / (2 / 3 * 0.01 + 1 / 3 * 0.99)
    assert m["precision_at_realistic_prevalence"] == pytest.approx(expected)
    nothing = operational_metrics(scores, labels, 5.0)
    assert nothing["precision"] == 0.0 and nothing["f1"] == 0.0


def test_ranking_and_oracle():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert ranking_metrics(scores, labels) == {"auroc": 1.0, "auprc": 1.0}
    tau = oracle_threshold(scores, labels)
    assert operational_metrics(scores, labels, tau)["f1"] == 1.0


def test_closed_world_counts_absent_classes():
    m = closed_world_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0]), n_classes=3)
    assert m["accuracy"] == 0.75
    assert m["macro_recall"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_evaluate_zero_day_uses_benign_tuning_flows_only():
    rng = np.random.default_rng(0)
    tune_benign, tune_attack = rng.normal(0, 1, 2000), rng.normal(6, 1, 2000)
    tune = np.concatenate([tune_benign, tune_attack])
    is_benign = np.concatenate([np.ones(2000), np.zeros(2000)])
    test = np.concatenate([rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)])
    labels = np.concatenate([np.zeros(2000), np.ones(2000)])
    result = evaluate_zero_day(test, labels, tune, is_benign, {"target_fpr": 0.05, "realistic_prevalence": 0.01})
    points = result["operating_points"]
    assert points["benign"]["threshold"] == pytest.approx(np.quantile(tune_benign, 0.95))
    assert points["mixture"]["threshold"] > points["benign"]["threshold"]
    assert points["benign"]["fpr"] == pytest.approx(0.05, abs=0.02)
    assert points["oracle"]["f1"] >= points["benign"]["f1"]
    assert result["n_calibration_benign"] == 2000


def test_wilcoxon_exact_known_values():
    base = np.zeros(10)
    all_positive = wilcoxon_exact(np.arange(1, 11), base)
    assert all_positive["p_value"] == pytest.approx(2 / 1024)  # smallest possible p with n = 10
    # W = 9: ranks 1, 3 and 5 negative. The exact p is 66/1024 = 0.0645 (the value in the old Table 5).
    diffs = np.array([-1, 2, -3, 4, -5, 6, 7, 8, 9, 10], dtype=float)
    result = wilcoxon_exact(diffs, base)
    assert result["statistic"] == 9 and result["p_value"] == pytest.approx(66 / 1024)
    assert result["p_value"] == pytest.approx(wilcoxon(diffs, method="exact").pvalue)
    assert wilcoxon_exact(base, base)["p_value"] == 1.0
    tied = wilcoxon_exact(np.array([1.0, 1.0, 1.0, -1.0, 0.0]), np.zeros(5))
    assert tied["n"] == 4 and tied["p_value"] == pytest.approx(10 / 16)
    with pytest.raises(ValueError):
        wilcoxon_exact(np.arange(30), np.zeros(30))


def test_benjamini_hochberg_known_values():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.02, 0.04, 0.04, 0.02])
    # Six tests: a raw p of 0.0098 can never adjust above 6 * 0.0098 = 0.0588.
    assert benjamini_hochberg([0.0098, 0.002, 0.002, 0.002, 0.002, 0.002]).max() <= 0.0588
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2])


def test_cliffs_delta():
    assert cliffs_delta([2, 3, 4], [0, 1]) == 1.0
    assert cliffs_delta([0, 1], [2, 3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert [delta_magnitude(d) for d in (0.1, 0.2, 0.4, -0.9)] == ["Negligible", "Small", "Medium", "Large"]


def test_latent_scorers_rank_far_points_higher():
    from zdssl.scoring import class_mahalanobis_min, knn_distance, latent_scores, mahalanobis_distance

    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 8))
    labels = np.repeat([0, 1], 250)
    ref[labels == 1] += 6
    near, far = rng.normal(size=(50, 8)), rng.normal(size=(50, 8)) + 20
    for score in (mahalanobis_distance(np.vstack([near, far]), ref[labels == 0]),
                  class_mahalanobis_min(np.vstack([near, far]), ref, labels),
                  knn_distance(np.vstack([near, far]), ref[labels == 0], k=5)):
        assert score[:50].max() < score[50:].min()
    known_attack = rng.normal(size=(20, 8)) + 6
    assert class_mahalanobis_min(known_attack, ref, labels).mean() < mahalanobis_distance(known_attack, ref[labels == 0]).mean()
    scores = latent_scores(np.vstack([near, far]), np.zeros(8), ref, labels, benign_index=0)
    assert all(v.shape == (100,) and np.isfinite(v).all() for v in scores.values())
    with pytest.raises(ValueError):
        latent_scores(near, np.zeros(8), ref[:1], labels[:1], benign_index=0)
