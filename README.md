# zdssl: self-supervised zero-day intrusion detection on CSE-CIC-IDS2018

A self-supervised tabular encoder for detecting an attack family that was never seen in training, evaluated against five baselines under label scarcity. This repository holds the code, the tests, and the full outputs of the run matrix.

The test is a chronological hold-out: models train on flows from 14 February to 1 March 2018 (14 known classes) and are tested on 2 March 2018, where the Bot class appears for the first time.

## Results

280 runs on a Colab Tesla T4: 6 models × 4 label budgets × 10 seeds, plus an objective ablation and a shortcut check. Every number below comes from [`paper1_results/results/runs.jsonl`](paper1_results/results/runs.jsonl); the tables are in [`paper1_results/report/`](paper1_results/report/) as CSV and LaTeX.

### 10% label budget, 10 seeds (mean ± sample std)

| Model | Zero-day AUROC | Zero-day AUPRC | Bot recall at τ | Closed-world Macro-F1 |
|---|---|---|---|---|
| **S4_E06, decoupled score (proposed)** | **0.9785 ± 0.0028** | **0.9423 ± 0.0095** | **0.9827 ± 0.0037** | 0.7001 ± 0.0234 |
| S4_E06, MSP score | 0.8747 ± 0.0822 | 0.7950 ± 0.1136 | 0.2230 ± 0.2554 | 0.7001 ± 0.0234 |
| S4_E05 (end-to-end fine-tuning) | 0.8818 ± 0.1086 | 0.8739 ± 0.0896 | 0.4892 ± 0.2030 | 0.7471 ± 0.0103 |
| Random Forest | 0.9528 ± 0.0207 | 0.9376 ± 0.0261 | 0.5809 ± 0.2426 | 0.8267 ± 0.0152 |
| XGBoost | 0.8352 ± 0.0946 | 0.7938 ± 0.1217 | 0.4561 ± 0.3128 | **0.8493 ± 0.0161** |
| Deep SVDD | 0.7993 ± 0.0277 | 0.8152 ± 0.0220 | 0.5095 ± 0.0041 | N/A |
| SCARF | 0.4672 ± 0.0933 | 0.6066 ± 0.1103 | 0.2756 ± 0.2394 | 0.6869 ± 0.0250 |

τ is the 95th percentile of each model's score on 10,800 benign validation flows (a 5% false-alarm target). The proposed model's observed false-positive rate on the test day is 0.0436 ± 0.0013. All six paired comparisons against the decoupled score are significant (exact Wilcoxon signed-rank, n = 10, Benjamini–Hochberg adjusted p ≤ 0.0039).

### Zero-day AUROC by label budget

| Model | 100% | 10% | 5% | 1% |
|---|---|---|---|---|
| S4_E06, decoupled | 0.9785 | 0.9785 | 0.9785 | 0.9785 |
| Random Forest | 0.9779 | 0.9528 | 0.9359 | 0.8512 |
| XGBoost | 0.9439 | 0.8352 | 0.7702 | 0.8161 |
| S4_E05 | 0.9154 | 0.8818 | 0.9033 | 0.8254 |
| S4_E06, MSP | 0.8931 | 0.8747 | 0.8487 | 0.8320 |
| Deep SVDD | 0.7224 | 0.7993 | 0.8348 | 0.9833 |
| SCARF | 0.6852 | 0.4672 | 0.4577 | 0.8044 |

The decoupled score is the distance from a flow's latent to a fixed centroid. It uses no labels, so it is the same at every budget by construction.

### Objective ablation (10% budget, 10 seeds)

| Pretraining objectives | Decoupled AUROC |
|---|---|
| Reconstruction | 0.6370 ± 0.0794 |
| Contrastive | 0.8077 ± 0.0706 |
| Reconstruction + contrastive | 0.9216 ± 0.0384 |
| Reconstruction + contrastive + novelty | 0.9785 ± 0.0028 |

## Limitations

- **Not best at every budget.** Random Forest ties the proposed model with 100% labels, and Deep SVDD is ahead at 1%.
- **Tree ensembles classify known attacks better** at every budget.
- **Bot is a favourable case.** When one known attack class is held out as a stand-in zero-day, the decoupled score averages 0.68 AUROC over four classes, against 0.83 for distance-to-benign scores ([`tune.jsonl`](paper1_results/results/tune.jsonl)).
- **One dataset, one zero-day family.** Nothing here shows the result transfers to other networks or attacks.
- **No shortcut effect was found.** Random Forest scores 0.9523 with port and protocol and 0.9528 without.

## Protocol

| Item | Value |
|---|---|
| Dataset | CSE-CIC-IDS2018, 16,232,943 flows after removing 59 repeated header lines |
| Zero-day class | Bot, 286,191 flows, all on 2018-03-02 |
| Features | 68 flow statistics; port, protocol, addresses and timestamp removed |
| Train (also the pretraining pool, labels hidden) | 160,526 flows |
| Validation | 20,066 tuning + 20,066 closed-world test |
| Locked test | 100,000 flows: 50,000 benign + 50,000 Bot |
| Label budgets | 160,526 / 16,052 / 8,025 / 1,606 flows |
| Seeds | 42 to 51 |

Leakage controls, all checked in code before any training ([`splits.py`](src/zdssl/splits.py)):
- Self-supervised pretraining sees the training split only.
- The scaler and the feature selection are fitted on the training split only.
- Alarm thresholds come from benign validation flows only; the test labels are read only for metrics.
- No row and no feature-identical flow is shared between the historical data and the test set.
- Hyperparameters were checked on stand-in zero-days from the historical days, never on the test day. The stated configuration was kept.

Split sizes, class counts and the check results are in [`manifest.json`](paper1_results/data/manifest.json).

## Repository layout

| Path | Contents |
|---|---|
| [`src/zdssl/`](src/zdssl/) | The package: data cleaning, splits, encoder, losses, pretraining, baselines, scoring, metrics, statistics, runner, report |
| [`configs/paper1.yaml`](configs/paper1.yaml) | Every hyperparameter and path |
| [`tests/`](tests/) | 38 tests on a synthetic dataset, including the leakage invariants |
| [`notebooks/`](notebooks/) | Thin Colab notebooks, generated by [`tools/build_notebooks.py`](tools/build_notebooks.py) |
| [`paper1_results/results/`](paper1_results/results/) | `runs.jsonl` (one line per run), `tune.jsonl`, run logs |
| [`paper1_results/report/`](paper1_results/report/) | All tables as CSV and LaTeX, and the budget figure |
| [`paper1_results/data/`](paper1_results/data/) | Split manifest and ingest statistics (not the dataset) |
| [`updated_report/`](updated_report/) | A two-page PDF summary of the results |

## Reproduce

Local checks (CPU, about a minute):

```bash
uv sync
uv run pytest
```

Full run (GPU; about 2 hours on a Colab T4, 7 GB download):

```bash
pip install awscli
PYTHONPATH=src python -m zdssl.cli data                       # download, clean, verify counts, build splits
PYTHONPATH=src python -m zdssl.cli run pilot                  # every model once
PYTHONPATH=src python -m zdssl.cli run p1 p2 ablation condA   # the full matrix, resumable
PYTHONPATH=src python -m zdssl.cli report                     # tables and figure
```

The data step reads the official public bucket `s3://cse-cic-ids2018` and stops if the row or Bot counts do not match. Outputs go to `paths.drive_root` in the config (a Google Drive folder by default; override with `--drive-root`). The run matrix skips runs already in `runs.jsonl`, so it can be restarted after a disconnect.

Model checkpoints are not in the repository; pretraining takes about 35 seconds per model and seed on a T4.
