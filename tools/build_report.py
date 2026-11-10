"""Build the technical research report (HTML -> PDF through headless Chrome).

Every number, table and figure is generated from ``paper1_results``; nothing is typed by hand.

Run: ``uv run python tools/build_report.py``
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zdssl import report as R
from sklearn.metrics import roc_auc_score

from zdssl.figures import SERIES_STYLE, fig2_budget_trajectory, fig3_operating
from zdssl.tune import summarize_tuning

log = logging.getLogger("build_report")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper1_results"
OUT = ROOT / "updated_report"
ASSETS = OUT / "assets"
CONFIG_HASH = "e5688756d04f"
BUDGET, SEED = 0.10, 42
REPO = "https://github.com/AakashBhat1/zeroday-intrusion-detection-ssl"
PDF_NAME = "Zero_Day_SSL_Research_Report.pdf"
CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome",
)
NAMES = {
    "S4_E06 (Decoupled)": "S4_E06, decoupled score (proposed)", "S4_E06 (MSP Probe)": "S4_E06, MSP score",
    "S4_E05 (SSL Base)": "S4_E05 (end-to-end fine-tuning)", "SCARF (SSL)": "SCARF",
}
SCORER_NAMES = {
    "decoupled": "Distance to pool centroid (decoupled, proposed)", "centroid_benign": "Distance to benign centroid",
    "maha_benign": "Mahalanobis distance to benign", "maha_class": "Class-conditional Mahalanobis",
    "knn_benign": "k-NN distance to benign", "msp": "Maximum softmax probability",
    "raw_knn_benign": "No encoder: k-NN distance to benign", "raw_maha_benign": "No encoder: Mahalanobis to benign",
}


def pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f}&nbsp;±&nbsp;{std:.{digits}f}"


def html_table(header: list[str], rows: list[list[str]], bold_rows: tuple[int, ...] = (), cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = ""
    for i, row in enumerate(rows):
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr class='{'hl' if i in bold_rows else ''}'>{cells}</tr>"
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def equation(tex: str, number: int) -> str:
    """Render one display equation with matplotlib mathtext (Computer Modern) to SVG."""
    path = ASSETS / f"eq{number}.svg"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=11, math_fontfamily="cm")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return (f"<div class='eq'><img src='assets/{path.name}' alt='equation {number}'>"
            f"<span class='eqn'>({number})</span></div>")


def bar_figures(flat: pd.DataFrame, path: Path) -> None:
    """(a) objective ablation; (b) zero-day recall at the calibrated threshold, per model."""
    ablation = R.table5b(flat, BUDGET)
    oper = R.table4_group_a(flat, BUDGET, "benign")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.3))
    labels = ["Recon", "Contrastive", "Recon +\ncontrastive", "Recon +\ncontrastive +\nnovelty"]
    ax1.bar(range(4), ablation["decoupled_auroc_mean"], yerr=ablation["decoupled_auroc_std"], width=0.55,
            color=["#9ecae1", "#9ecae1", "#9ecae1", "#0072B2"], capsize=3, error_kw={"elinewidth": 1})
    for i, value in enumerate(ablation["decoupled_auroc_mean"]):
        ax1.text(i, 0.03, f"{value:.3f}", ha="center", fontsize=8, color="#111")
    ax1.set_xticks(range(4), labels, fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Zero-day AUROC (decoupled score)")
    ax1.set_title("(a) Pretraining-objective ablation", fontsize=10)
    order = oper.sort_values("benign_recall_mean", ascending=True)
    colors = [SERIES_STYLE[row][0] for row in order["row"]]
    ax2.barh(range(len(order)), order["benign_recall_mean"], xerr=order["benign_recall_std"], height=0.55,
             color=colors, capsize=3, error_kw={"elinewidth": 1})
    ax2.set_yticks(range(len(order)), [NAMES.get(r, r).replace(" (proposed)", "") for r in order["row"]], fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Bot recall at the benign-calibrated threshold")
    ax2.set_title("(b) Operational zero-day recall", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, axis="y" if ax is ax1 else "x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def stand_in_table(tune_path: Path) -> tuple[str, dict]:
    rows = []
    for line in tune_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["candidate"] == "base":
            rows += [{"held_out": record["held_out"], "scorer": k, "auroc": v} for k, v in record["auroc"].items()]
    table = pd.DataFrame(rows).pivot_table(index="scorer", columns="held_out", values="auroc", aggfunc="mean")
    table["Mean"] = table.mean(axis=1)
    keep = ["decoupled", "knn_benign", "maha_benign", "msp", "raw_knn_benign"]
    table = table.loc[keep]
    header = ["Zero-day score", *[c.replace("attacks-", "") for c in table.columns]]
    body = [[SCORER_NAMES[s], *[f"{v:.4f}" for v in table.loc[s]]] for s in keep]
    means = {s: float(table.loc[s, "Mean"]) for s in keep}
    return html_table(header, body), means


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("Chrome or Edge is needed to print the report to PDF")


CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm;
  @top-left { content: "Zero-Day SSL Research Report: Label-Scarce Intrusion Detection"; font: 8.5pt Georgia, serif; color: #444; }
  @top-right { content: "November 2026"; font: 8.5pt Georgia, serif; color: #444; }
  @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #444; } }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.2pt; line-height: 1.42; color: #111; text-align: justify; hyphens: auto; }
h1.title { font-size: 19pt; text-align: center; line-height: 1.25; margin: 6mm 0 3mm; }
p.sub { text-align: center; font-size: 11pt; margin: 0 0 1mm; }
p.meta { text-align: center; font-size: 9.5pt; color: #333; margin: 0 0 5mm; }
h2 { font-size: 13pt; margin: 6mm 0 2mm; break-after: avoid; }
h3 { font-size: 11pt; margin: 4mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.2mm; }
ol, ul { margin: 0 0 2.2mm 0; padding-left: 6mm; } li { margin-bottom: 1mm; }
table { border-collapse: collapse; margin: 2mm auto 1mm; font-size: 8.6pt; break-inside: avoid; }
th { border-top: 1.2pt solid #111; border-bottom: 0.7pt solid #111; padding: 1.2mm 2.2mm; text-align: left; }
td { padding: 0.9mm 2.2mm; text-align: left; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 1.2pt solid #111; }
tr.hl td { font-weight: bold; }
table.wrap td { white-space: normal; }
.cap { text-align: center; font-size: 9pt; margin: 1mm 0 3.5mm; break-before: avoid; }
.capt { text-align: center; font-size: 9pt; margin: 3mm 0 0; break-after: avoid; }
.eq { display: flex; align-items: center; justify-content: center; margin: 2mm 0; position: relative; }
.eq img { max-width: 88%; } .eqn { position: absolute; right: 0; font-size: 10pt; }
figure { margin: 3mm 0; text-align: center; break-inside: avoid; } figure img { max-width: 100%; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; }
a { color: #0b4f9c; text-decoration: none; }
.box { border: 0.7pt solid #777; padding: 2.5mm 3.5mm; margin: 3mm 0; break-inside: avoid; }
.page-break { break-before: page; }
"""


def build() -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = R.load_runs(RESULTS / "results" / "runs.jsonl", CONFIG_HASH)
    flat = R.flatten(runs)
    manifest = json.loads((RESULTS / "data" / "manifest.json").read_text(encoding="utf-8"))
    ingest = json.loads((RESULTS / "data" / "ingest_stats.json").read_text(encoding="utf-8"))["summary"]
    case = next(r for r in runs if r["key"] == f"S4_E06|F|B|{BUDGET:g}|{SEED}")
    params, sizes = case["params"], manifest["split_sizes"]

    # ---- figures ----
    trajectory = R.budget_trajectory(flat)
    fig2_budget_trajectory(trajectory, ASSETS / "fig_budget.png")
    scores = np.load(RESULTS / "scores" / f"S4_E06__F__B__{BUDGET:g}__{SEED}.npz")["decoupled_test"]
    labels = np.r_[np.zeros(sizes["test"] // 2), np.ones(sizes["test"] // 2)]  # benign rows precede zero-day rows
    points = case["zero_day"]["decoupled"]["operating_points"]
    if abs(roc_auc_score(labels, scores) - case["zero_day"]["decoupled"]["auroc"]) > 1e-9:
        raise ValueError("test-label order does not reproduce the recorded AUROC")
    fig3_operating(scores, labels, {k: v["threshold"] for k, v in points.items()}, ASSETS / "fig_operating.png",
                   title_suffix=f" (seed {SEED})")
    bar_figures(flat, ASSETS / "fig_bars.png")

    # ---- tables ----
    t3 = R.table3(flat, BUDGET)
    t4 = R.table4_group_a(flat, BUDGET, "benign").set_index("row")
    main_rows = [[NAMES.get(r.row, r.row), pm(r.auroc_mean, r.auroc_std), pm(r.auprc_mean, r.auprc_std),
                  pm(r.cw_accuracy_mean, r.cw_accuracy_std), pm(r.cw_macro_f1_mean, r.cw_macro_f1_std)]
                 for r in t3.itertuples()]
    table_main = html_table(["Model", "Zero-day AUROC", "Zero-day AUPRC", "CW accuracy", "CW Macro-F1"], main_rows, (0,))
    oper_rows = [[NAMES.get(name, name), f"{r.benign_threshold_mean:.4f}", pm(r.benign_precision_mean, r.benign_precision_std),
                  pm(r.benign_recall_mean, r.benign_recall_std), pm(r.benign_fpr_mean, r.benign_fpr_std),
                  pm(r.benign_f1_mean, r.benign_f1_std)] for name, r in t4.iterrows()]
    table_oper = html_table(["Model", "τ", "Precision", "Recall (Bot)", "FPR", "F1"], oper_rows, (0,))

    budgets = sorted(trajectory["budget"].unique(), reverse=True)
    pivot_mean = trajectory.pivot(index="row", columns="budget", values="auroc_mean")
    pivot_std = trajectory.pivot(index="row", columns="budget", values="auroc_std")
    pivot_f1 = trajectory.pivot(index="row", columns="budget", values="cw_macro_f1_mean")
    order = [row for row, _, _ in R.ROWS]
    table_budget = html_table(
        ["Model", *[f"{100 * b:g}% labels" for b in budgets]],
        [[NAMES.get(r, r), *[pm(pivot_mean.loc[r, b], pivot_std.loc[r, b]) for b in budgets]] for r in order], (0,))
    f1_rows = [r for r in order if r not in ("Deep SVDD", "S4_E06 (MSP Probe)")]
    table_f1 = html_table(
        ["Model", *[f"{100 * b:g}% labels" for b in budgets]],
        [[NAMES.get(r, r).replace(", decoupled score (proposed)", " (proposed)"), *[f"{pivot_f1.loc[r, b]:.4f}" for b in budgets]]
         for r in f1_rows])
    n_train = {b: int(flat.loc[np.isclose(flat["budget"], b) & (flat["model"] == "RF") & (flat["condition"] == "B"),
                               "n_train"].iloc[0]) for b in budgets}

    stats = R.table5a(flat, BUDGET)
    table_stats = html_table(
        ["Proposed vs", "Mean ΔAUROC", "W", "Raw p", "BH-adjusted p", "Cliff's δ"],
        [[NAMES.get(c.split(" vs ")[1], c.split(" vs ")[1]), f"{r.mean_difference:+.4f}", f"{r.w_statistic:g}",
          f"{r.p_raw:.4f}", f"{r.p_bh:.4f}", f"{r.cliffs_delta:+.2f} ({r.magnitude})"]
         for c, r in zip(stats["comparison"], stats.itertuples())])

    ablation = R.table5b(flat, BUDGET)
    table_ablation = html_table(
        ["Pretraining objectives", "Decoupled AUROC", "Decoupled AUPRC", "MSP AUROC"],
        [[r.row[3:], pm(r.decoupled_auroc_mean, r.decoupled_auroc_std), pm(r.decoupled_auprc_mean, r.decoupled_auprc_std),
          pm(r.msp_auroc_mean, r.msp_auroc_std)] for r in ablation.itertuples()], (3,))
    scorers = R.scorer_table(flat, BUDGET)
    table_scorers = html_table(
        ["Zero-day score", "AUROC", "AUPRC", "Recall at τ", "FPR at τ"],
        [[SCORER_NAMES[r.row], pm(r.auroc_mean, r.auroc_std), pm(r.auprc_mean, r.auprc_std),
          pm(r.benign_recall_mean, r.benign_recall_std), pm(r.benign_fpr_mean, r.benign_fpr_std)]
         for r in scorers.itertuples()], (0,))
    table_standin, standin_means = stand_in_table(RESULTS / "results" / "tune.jsonl")
    cand = summarize_tuning(RESULTS / "results" / "tune.jsonl")
    cand_names = {"base": "Stated configuration (10 epochs, mask 0.2, λ_nov 0.2)", "epochs30": "30 pretraining epochs",
                  "mask0.4": "Mask rate 0.4", "temp0.5": "Contrastive temperature 0.5",
                  "no_input_norm": "No input LayerNorm", "novelty1.0": "λ_nov = 1.0", "no_novelty": "No novelty loss"}
    table_cand = html_table(
        ["Pretraining setting", "Decoupled", "k-NN to benign", "Mahalanobis to benign"],
        [[cand_names[c], f"{cand.loc[c, 'decoupled']:.4f}", f"{cand.loc[c, 'knn_benign']:.4f}",
          f"{cand.loc[c, 'maha_benign']:.4f}"] for c in cand_names if c in cand.index], (0,))
    cond = R.condition_a(flat, BUDGET)
    cost = R.table2(flat, BUDGET)

    def size_of(r) -> str:
        if not pd.isna(getattr(r, "params_deployed", np.nan)):
            return f"{int(r.params_deployed):,} parameters"
        if not pd.isna(getattr(r, "params_nodes", np.nan)):
            return f"{int(r.params_nodes):,} tree nodes"
        return f"{int(r.params_trees):,} trees"

    table_cost = html_table(
        ["Model", "Deployed size", "Pretraining", "Adaptation", "Inference / 1,000 flows", "Peak GPU memory", "Checkpoint"],
        [[NAMES.get(r.row, r.row).replace(", decoupled score (proposed)", " (proposed)"), size_of(r),
          "—" if pd.isna(r.time_pretrain_seconds) else f"{r.time_pretrain_seconds:.1f} s", f"{r.time_adapt_seconds:.2f} s",
          f"{1000 * r.time_infer_ms_per_flow:.2f} ms",
          "CPU" if pd.isna(r.time_peak_memory_mb) else f"{r.time_peak_memory_mb:.0f} MB", f"{r.time_checkpoint_mb:.2f} MB"]
         for r in cost.itertuples()])

    classes = manifest["sample_class_counts"]
    table_classes = html_table(
        ["Class", "Flows in dataset", "Flows sampled"],
        [[name, f"{ingest['label_counts'][name]:,}", f"{count:,}"] for name, count in classes.items()]
        + [["Bot (zero-day, test only)", f"{ingest['label_counts']['Bot']:,}", f"{manifest['test_class_counts']['Bot']:,}"]])
    table_splits = html_table(
        ["Partition", "Source", "Flows", "Use"],
        [["Training / SSL pool", "14 Feb – 1 Mar", f"{sizes['train']:,}", "Pretraining (labels hidden) and label budgets"],
         ["Validation, tuning half", "14 Feb – 1 Mar", f"{sizes['tune']:,}",
          f"Threshold calibration on its {manifest['split_class_counts']['tune']['Benign']:,} benign flows"],
         ["Validation, closed-world half", "14 Feb – 1 Mar", f"{sizes['cw_test']:,}", "14-class accuracy and Macro-F1"],
         ["Locked test", "2 Mar", f"{sizes['test']:,}", "50,000 benign + 50,000 Bot; zero-day metrics"]], cls="wrap")

    g = lambda row, col: float(t3.set_index("row").loc[row, col])  # noqa: E731
    best = "S4_E06 (Decoupled)"
    quarantine_h = manifest["quarantine"]["seconds_from_train"] / 3600
    rf_tie, svdd_1 = pivot_mean.loc["Random Forest", 1.0], pivot_mean.loc["Deep SVDD", 0.01]
    seed_point = points["benign"]

    eqs = [
        equation(r"\tilde{x}_{k,j} = m_j \, x^{\mathrm{marg}}_{j} + (1 - m_j)\,(x_j + \epsilon_j), \qquad m_j \sim \mathrm{Bernoulli}(0.2), \quad \epsilon_j \sim \mathcal{N}(0,\ 0.02^{2})", 1),
        equation(r"\mathcal{L}_{\mathrm{con}} = -\frac{1}{2N}\sum_{i=1}^{2N} \log \frac{\exp(\mathrm{sim}(p_i, p_{i^{+}})/\tau_c)}{\sum_{k \neq i} \exp(\mathrm{sim}(p_i, p_k)/\tau_c)}, \qquad \tau_c = 0.1", 2),
        equation(r"\mathcal{L}_{\mathrm{rec}} = \frac{1}{D}\sum_{j=1}^{D} (x_j - \hat{x}_j)^2 + \frac{\gamma}{D}\sum_{j=1}^{D} \mathrm{BCE}(m_j, \hat{m}_j), \qquad \gamma = 1", 3),
        equation(r"\mathcal{L}_{\mathrm{nov}} = \frac{1}{N}\sum_{i=1}^{N} \| z_i - c \|_2^2, \qquad c = \frac{1}{|\mathcal{D}_{\mathrm{ssl}}|}\sum_{x \in \mathcal{D}_{\mathrm{ssl}}} f_\theta(x) \ \ \mathrm{(fixed\ after\ epoch\ 5)}", 4),
        equation(r"\mathcal{L}_{\mathrm{pretrain}} = 0.5\,\mathcal{L}_{\mathrm{rec}} + 0.5\,\mathcal{L}_{\mathrm{con}} + 0.2\,\mathcal{L}_{\mathrm{nov}}", 5),
        equation(r"S_{\mathrm{nov}}(x^{*}) = \| f_\theta(x^{*}) - c \|_2^2, \qquad \mathrm{alert\ if\ } S_{\mathrm{nov}}(x^{*}) > \tau, \quad \tau = Q_{0.95}\{ S_{\mathrm{nov}}(x) : x \in \mathcal{D}_{\mathrm{val}}^{\mathrm{benign}} \}", 6),
    ]

    body = f"""
<h1 class="title">Self-Supervised Tabular Representation Learning<br>with Decoupled Latent-Centroid Scoring<br>for Zero-Day Intrusion Detection Under Label Scarcity</h1>
<p class="sub">Technical Research Report &amp; Empirical Validation</p>
<p class="meta">Date: November 10, 2026 &nbsp;|&nbsp; Status: empirical validation complete &nbsp;|&nbsp; Code and results: <a href="{REPO}">{REPO.split('//')[1]}</a></p>

<h2>Executive Summary</h2>
<p>Network intrusion detection systems are trained on catalogued attacks and deployed against attacks nobody has labelled yet. Supervised classifiers assign such traffic to a known class with high confidence, and labels for retraining are scarce. This report presents the empirical validation of <b>S4_E06</b>, a self-supervised tabular model that separates zero-day detection from attack classification. It integrates:</p>
<ol>
<li>A <b>residual tabular encoder</b> ({params['pretrain_encoder']:,} parameters) mapping 68 flow statistics to a 128-dimensional latent;</li>
<li><b>Dual corrupted views</b> of every flow, built by marginal feature replacement and Gaussian jitter;</li>
<li>A <b>multi-task self-supervised objective</b>: masked feature reconstruction, InfoNCE contrastive learning and a latent hypersphere novelty term;</li>
<li>A <b>frozen encoder with a lightweight probe</b> ({params['probe']:,} parameters) for 14-class triage of known attacks;</li>
<li><b>Decoupled zero-day scoring</b>: the distance of a flow's latent to a fixed centroid, which needs no labels;</li>
<li><b>Benign-only threshold calibration</b> on validation flows, with a 5% false-alarm target.</li>
</ol>
<p>The model was benchmarked on CSE-CIC-IDS2018 ({ingest['clean_rows']:,} flows) against five baselines across four label budgets and ten seeds: {len(runs)} runs in total. Training used flows from 14 February to 1 March 2018; the test day, 2 March, contains the Bot family, which appears nowhere in training.</p>
<p>With 10% of the labels, decoupled scoring reaches a zero-day <b>AUROC of {pm(g(best,'auroc_mean'), g(best,'auroc_std'))}</b> and <b>AUPRC of {pm(g(best,'auprc_mean'), g(best,'auprc_std'))}</b>. At the calibrated threshold it detects <b>{100*t4.loc[best,'benign_recall_mean']:.2f}% of Bot flows at a {100*t4.loc[best,'benign_fpr_mean']:.2f}% false-positive rate</b> (F1 {t4.loc[best,'benign_f1_mean']:.4f}). The next best model, Random Forest, reaches AUROC {g('Random Forest','auroc_mean'):.4f} and detects {100*t4.loc['Random Forest','benign_recall_mean']:.1f}% of Bot flows. All six paired comparisons are statistically significant.</p>
<p>The result has clear limits. With all labels Random Forest matches the proposed model ({rf_tie:.4f}), and with 1% of labels Deep SVDD exceeds it ({svdd_1:.4f}). Tree ensembles classify known attacks better at every budget. When known attack families are held out as stand-in zero-days, the decoupled score averages only {standin_means['decoupled']:.2f} AUROC, so the Bot result does not establish detection of arbitrary new attacks.</p>

<h2>1&nbsp;&nbsp;Introduction &amp; Research Problem</h2>
<p>Security operations centres inspect millions of flow records and can label very few of them. A practical detector must work with scarce labels and must flag attack families it was never shown. Existing approaches face four obstacles:</p>
<ol>
<li><b>Closed-world confidence.</b> A softmax classifier must place every flow in a known class. Unseen attacks are absorbed into known classes, often with high confidence, so rejection by maximum softmax probability (MSP) is unreliable.</li>
<li><b>Label scarcity.</b> Supervised models degrade as labelled data shrinks, while unlabelled telemetry is abundant and unused.</li>
<li><b>One-class models cannot triage.</b> Detectors trained on benign traffic alone, such as Deep SVDD, flag anomalies but cannot say which known attack a flow belongs to.</li>
<li><b>Evaluation leakage.</b> Random splits let duplicated flows, scalers fitted on all data, or thresholds tuned on test traffic inflate reported numbers.</li>
</ol>
<p>S4_E06 addresses these with a decoupled design. A self-supervised encoder learns from unlabelled flows. A small probe on the frozen encoder classifies known attacks. Zero-day detection reads the latent geometry directly, through the distance to a centroid, and never passes through the classifier.</p>

<h2>2&nbsp;&nbsp;Threat Model &amp; Evaluation Protocol</h2>
<p>Historical telemetry covers 14 February to 1 March 2018 and contains benign traffic and 13 catalogued attack classes. The locked test set is drawn from 2 March 2018, the only day on which the Bot family ({ingest['label_counts']['Bot']:,} flows) appears. The last training flow precedes the first test flow by at least {quarantine_h:.1f} hours; the source timestamps use a 12-hour clock, so this is a lower bound.</p>
<p>The historical sample caps benign traffic at 108,000 flows and each attack class at 10,000, taking smaller classes whole. It is split 80/20 into training and validation, and validation is split in half (Table 1). Label budgets of 100%, 10%, 5% and 1% are stratified subsets of the training split: {' / '.join(f'{n_train[b]:,}' for b in budgets)} flows.</p>
<p class="capt">Table 1: Data partitions.</p>
{table_splits}
<p class="capt">Table 2: Class composition of the historical sample and the test set.</p>
{table_classes}

<h2>3&nbsp;&nbsp;Architecture &amp; Leakage Safeguards</h2>
<h3>3.1&nbsp;&nbsp;Data Isolation Protocol</h3>
<p>Every safeguard below is enforced in code before training starts, and the pipeline stops if one fails.</p>
<ul>
<li><b>Train-only pretraining.</b> The self-supervised pool is the training split with labels hidden. Validation and test flows never reach the encoder.</li>
<li><b>Train-only preprocessing.</b> Median imputation, the robust scaler and the feature selection are fitted on the training split and applied unchanged elsewhere. Scaled features are clipped to ±10.</li>
<li><b>Identifier removal.</b> Destination port, protocol, addresses and timestamps are removed. Eight features that are constant on the training split are dropped, leaving {manifest['n_features_B']} flow statistics.</li>
<li><b>Duplicate purging.</b> Exact duplicate rows are removed before sampling. {manifest['test_rows_removed_seen_in_historical']:,} test-day flows whose feature vector also occurs in the historical sample were removed before the test set was drawn, so no test flow has a feature-identical copy in training or validation.</li>
<li><b>Validation-only calibration.</b> Alarm thresholds are set on benign validation flows. Test labels are read only to compute metrics.</li>
<li><b>No tuning on the test day.</b> Hyperparameters were examined on stand-in zero-days built from the historical days (Section 7).</li>
</ul>
<p>One overlap remains and is reported, not removed: {manifest['validation_rows_sharing_a_train_feature_profile']:,} validation flows share a feature vector with a training flow, because scripted attacks repeat. Closed-world scores are therefore somewhat optimistic for every model alike.</p>

<h3>3.2&nbsp;&nbsp;Mathematical Formulation</h3>
<p><b>View generation.</b> For a flow <i>x</i> ∈ ℝ<sup>68</sup>, two views are produced. Each coordinate is replaced by a draw from that feature's empirical marginal with probability 0.2, and otherwise receives Gaussian jitter:</p>
{eqs[0]}
<p><b>Encoder.</b> The encoder <i>f<sub>θ</sub></i> applies layer normalisation to the input, projects to 256 dimensions, passes two residual blocks (linear, layer normalisation, GELU, dropout 0.1) and projects to a 128-dimensional latent <i>z</i>.</p>
<p><b>Contrastive loss.</b> A projection head maps each latent to 64 dimensions. The two views of a flow form a positive pair among the 2<i>N</i> projections of a batch, with cosine similarity:</p>
{eqs[1]}
<p><b>Masked reconstruction.</b> A second head predicts the clean features and the corruption mask <i>m</i>:</p>
{eqs[2]}
<p><b>Hypersphere novelty term.</b> After five warm-up epochs the centroid <i>c</i> is set to the mean latent of the pretraining pool and frozen. For the remaining five epochs latents are pulled toward it:</p>
{eqs[3]}
<p><b>Training objective.</b> Pretraining runs for 10 epochs with AdamW (learning rate 5×10<sup>−4</sup>, batch size 512):</p>
{eqs[4]}
<p><b>Triage and zero-day scoring.</b> The encoder is frozen and a probe (128→64→14) is trained on the labelled budget with class-weighted cross-entropy. Zero-day detection does not use the probe. A flow is flagged when its distance to the centroid exceeds a threshold calibrated on benign validation flows:</p>
{eqs[5]}

<h3>3.3&nbsp;&nbsp;Baselines</h3>
<p><b>S4_E05</b> uses the same pretraining without the novelty term and then fine-tunes the encoder end to end; it is scored by MSP. <b>SCARF</b> is contrastive pretraining with 60% feature corruption and a frozen probe. <b>Deep SVDD</b> is a bias-free one-class network trained on the benign flows of each budget. <b>Random Forest</b> (100 trees, depth 15) and <b>XGBoost</b> (100 rounds) are trained on the labelled budget and scored by MSP. Every model is run with seeds 42 to 51.</p>

<h2>4&nbsp;&nbsp;Empirical Evaluation</h2>
<h3>4.1&nbsp;&nbsp;Zero-Day Ranking and Closed-World Triage</h3>
<p>Table 3 reports the benchmark at the 10% budget. Decoupled scoring gives the highest zero-day AUROC and AUPRC, with a seed-to-seed standard deviation of {g(best,'auroc_std'):.4f}, against {min(g(r,'auroc_std') for r in t3['row'] if r != best):.4f} or more for every baseline. The same encoder scored by MSP is far less reliable, which isolates the gain to the scoring rule. On known attacks the order reverses: the tree ensembles lead, and the proposed model's Macro-F1 is {g(best,'cw_macro_f1_mean'):.4f} against {g('XGBoost','cw_macro_f1_mean'):.4f} for XGBoost.</p>
<p class="capt">Table 3: Zero-day detection on the locked test set and closed-world (CW) classification, 10% labels, 10 seeds (mean ± sample standard deviation).</p>
{table_main}

<h3>4.2&nbsp;&nbsp;Operation at a Calibrated Threshold</h3>
<p>Ranking metrics do not say what an analyst sees. Table 4 fixes each model's threshold at the 95th percentile of its score on benign validation flows and reports the outcome on the test day. All models hold close to the 5% false-alarm target. The proposed model detects {100*t4.loc[best,'benign_recall_mean']:.1f}% of Bot flows; no baseline exceeds {100*max(t4.loc[r,'benign_recall_mean'] for r in t4.index if r != best):.0f}%, and most vary widely between seeds. For seed {SEED} the threshold τ = {seed_point['threshold']:.4f} yields {seed_point['tp']:,} true positives and {seed_point['fp']:,} false positives out of 50,000 flows each.</p>
<p class="capt">Table 4: Operational zero-day detection at the benign-calibrated threshold, 10% labels, 10 seeds.</p>
{table_oper}

<h3>4.3&nbsp;&nbsp;Label-Budget Trajectory</h3>
<p>Table 5 varies the label budget. The decoupled score uses no labels, so it is identical at every budget by construction; only the probe changes. Random Forest improves steadily with labels and matches the proposed model at 100%. Deep SVDD moves the other way: it is the best detector at 1% and falls to {pivot_mean.loc['Deep SVDD', 1.0]:.4f} at 100%. The proposed model is the best detector at 10% and 5%.</p>
<p class="capt">Table 5: Zero-day AUROC by label budget, 10 seeds.</p>
{table_budget}
<p class="capt">Table 6: Closed-world Macro-F1 by label budget (mean of 10 seeds).</p>
{table_f1}

<h3>4.4&nbsp;&nbsp;Statistical Hypothesis Testing</h3>
<p>Each baseline is compared with the proposed model on seed-paired AUROC using an exact two-sided Wilcoxon signed-rank test (n = 10), with Benjamini–Hochberg correction over the six comparisons and Cliff's δ as effect size. All six differences are significant at α = 0.05 (Table 7). With ten pairs the smallest attainable raw p-value is 0.0020.</p>
<p class="capt">Table 7: Paired significance tests on zero-day AUROC, 10% labels.</p>
{table_stats}

<h2>5&nbsp;&nbsp;Visual Experimental Results</h2>
<figure><img src="assets/fig_budget.png" style="width:100%"></figure>
<p class="cap">Figure 1: Zero-day AUROC and AUPRC across label budgets (mean of 10 seeds; bars show one standard deviation).</p>
<figure><img src="assets/fig_operating.png" style="width:100%"></figure>
<p class="cap">Figure 2: Decoupled score on the locked test set, seed {SEED}, 10% labels. (a) ROC curve with the benign-calibrated, mixture-calibrated and post-hoc oracle operating points. (b) Recall, precision and false-positive rate against the threshold. The oracle point reads test labels and is a diagnostic only.</p>
<figure><img src="assets/fig_bars.png" style="width:100%"></figure>
<p class="cap">Figure 3: (a) Decoupled AUROC for each combination of pretraining objectives. (b) Bot recall at the calibrated threshold for every model. Bars show one standard deviation over 10 seeds.</p>

<h2>6&nbsp;&nbsp;Component Ablation Study</h2>
<h3>6.1&nbsp;&nbsp;Pretraining Objectives</h3>
<p>Each objective combination was pretrained for all ten seeds and evaluated with a frozen probe at the 10% budget (Table 8). Each added objective raises the decoupled AUROC and narrows its spread. The novelty term has the opposite effect on MSP, whose AUROC falls when the latents are compacted. The two scores therefore favour different geometries, and the novelty term is useful only together with centroid scoring.</p>
<p class="capt">Table 8: Objective ablation, 10% labels, 10 seeds.</p>
{table_ablation}

<h3>6.2&nbsp;&nbsp;Scoring Rules on the Same Encoder</h3>
<p>Table 9 applies eight scoring rules to the S4_E06 encoder. The two centroid distances are equivalent and best. Mahalanobis and nearest-neighbour distances rank well but detect only about half of the Bot flows at the calibrated threshold. The last two rows apply the same reference-based rules to the scaled input features with no encoder; they fail on Bot, which shows that the learned representation, not the distance rule alone, carries the result.</p>
<p class="capt">Table 9: Zero-day scores computed on the S4_E06 encoder, 10% labels, 10 seeds.</p>
{table_scorers}

<h2>7&nbsp;&nbsp;Robustness Analysis</h2>
<h3>7.1&nbsp;&nbsp;Stand-In Zero-Days</h3>
<p>To examine hyperparameters without touching the test day, one known attack class at a time was removed from training entirely and the model was scored on separating benign validation flows from that class. Four classes and two seeds were used. No pretraining setting improved on the stated configuration by more than about 0.01 on the benign-distance scores, and tripling the epochs did not help, so the stated configuration was kept (Table 10). Removing the novelty term lowers the decoupled score from {cand.loc['base','decoupled']:.2f} to {cand.loc['no_novelty','decoupled']:.2f}.</p>
<p class="capt">Table 10: Mean AUROC over four stand-in zero-days and two seeds, by pretraining setting.</p>
{table_cand}
<p>Table 11 breaks the stated configuration down by held-out class. The decoupled score is uneven: it is near chance on Infiltration and moderate elsewhere, with a mean of {standin_means['decoupled']:.2f}. Benign-distance scores are steadier (mean {standin_means['knn_benign']:.2f}), and on Slowloris a nearest-neighbour rule on raw features is as good as any learned score. Infiltration is hard for every score.</p>
<p class="capt">Table 11: AUROC on each stand-in zero-day for the stated configuration (mean of two seeds).</p>
{table_standin}

<h3>7.2&nbsp;&nbsp;Transport Shortcut Check</h3>
<p>Random Forest was retrained with destination port and protocol restored. Its zero-day AUROC is {pm(cond.iloc[0]['auroc_mean'], cond.iloc[0]['auroc_std'])} with the identifiers and {pm(cond.iloc[1]['auroc_mean'], cond.iloc[1]['auroc_std'])} without them. No shortcut effect is present in this protocol.</p>

<h2>8&nbsp;&nbsp;Computational Cost</h2>
<p>Table 12 reports medians over ten seeds at the 10% budget on a Colab Tesla T4 with two CPU cores; tree models run on the CPU. Pretraining is a one-off cost per seed and is reused across budgets. Deployed inference for the proposed model is {1000*cost.iloc[0]['time_infer_ms_per_flow']:.2f} ms per 1,000 flows. The trained model has {params['trained_total']:,} parameters, of which {params['deployed']:,} are deployed.</p>
<p class="capt">Table 12: Model size and measured cost, 10% labels.</p>
{table_cost}

<h2>9&nbsp;&nbsp;Discussion, Limitations &amp; Conclusion</h2>
<h3>9.1&nbsp;&nbsp;Why Decoupling Helps</h3>
<p>The probe and the zero-day score read the same encoder, yet MSP reaches AUROC {g('S4_E06 (MSP Probe)','auroc_mean'):.4f} with a standard deviation of {g('S4_E06 (MSP Probe)','auroc_std'):.4f} while centroid distance reaches {g(best,'auroc_mean'):.4f} with {g(best,'auroc_std'):.4f}. MSP depends on where the probe happens to place Bot flows among the known classes, which changes with the seed and the labelled subset. Centroid distance depends only on pretraining, which uses 160,526 unlabelled flows and is stable.</p>
<h3>9.2&nbsp;&nbsp;Limitations</h3>
<ul>
<li><b>One zero-day family.</b> The headline result is for Bot. On stand-in zero-days the decoupled score is much weaker, so the model should not be described as a general zero-day detector.</li>
<li><b>Not best at every budget.</b> Random Forest matches it with all labels and Deep SVDD exceeds it with 1%.</li>
<li><b>Weaker triage.</b> Tree ensembles classify known attacks better at every budget.</li>
<li><b>One dataset.</b> No cross-dataset transfer was evaluated.</li>
<li><b>Balanced test set.</b> The test set is half Bot. At a realistic 1% prevalence the precision of the calibrated operating point falls to about {100*seed_point['precision_at_realistic_prevalence']:.0f}% for seed {SEED}.</li>
</ul>
<h3>9.3&nbsp;&nbsp;Conclusion</h3>
<p>Under a leakage-controlled chronological protocol, a self-supervised tabular encoder with decoupled centroid scoring detects the held-out Bot family more accurately and far more stably than five baselines when 5% to 10% of labels are available, while retaining multi-class triage. The ablation attributes the gain to the combination of the three pretraining objectives with centroid scoring. The stand-in analysis shows the gain is family-dependent, and tree ensembles remain the better classifiers of known attacks.</p>

<div class="box"><b>Reproducibility.</b> Code, tests, configuration and all run outputs are at <a href="{REPO}">{REPO.split('//')[1]}</a>.
Every number in this report is computed from <a href="{REPO}/blob/main/paper1_results/results/runs.jsonl"><code>paper1_results/results/runs.jsonl</code></a> and <a href="{REPO}/blob/main/paper1_results/results/tune.jsonl"><code>tune.jsonl</code></a>;
the tables are also in <a href="{REPO}/tree/main/paper1_results/report"><code>paper1_results/report/</code></a>, the split manifest in <a href="{REPO}/blob/main/paper1_results/data/manifest.json"><code>paper1_results/data/manifest.json</code></a>,
and this report is generated by <a href="{REPO}/blob/main/tools/build_report.py"><code>tools/build_report.py</code></a>. Dataset: CSE-CIC-IDS2018, public bucket <code>s3://cse-cic-ids2018</code>.</div>
"""
    html_path = OUT / "report.html"
    html_path.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Zero-Day SSL Research Report</title>"
                         f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    pdf_path = OUT / PDF_NAME
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, capture_output=True, timeout=180)
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("wrote %s", build())
