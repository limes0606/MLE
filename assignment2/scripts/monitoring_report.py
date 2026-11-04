"""
Build the monitoring visuals + a summary table from the gold monitoring table.
Usage: python3 monitoring_report.py --snapshotdate 2025-06-01   (runs once, on the last run date)

Outputs in datamart/gold/model_monitoring/charts/ :
  1_performance.png        monthly AUC (95% CI), rolling 3-month AUC, baseline and alert line
  2_stability_psi.png      score PSI and worst feature PSI vs thresholds
  3_feature_psi_heatmap.png  PSI of every feature, every month
  4_clickstream_coverage.png data-quality alert (no clickstream rows for new customers)
  5_calibration.png        mean predicted default probability vs actual bad rate
  7_challenger.png         challenger vs champion lift on unseen cohorts
  8_cutoff_tradeoff.png    who is flagged at each cutoff, and estimated value under margin scenarios (+ cutoff_tradeoff.csv)
  8b_cutoff_gain.png       simple view: losses avoided against margin lost at each cutoff, one panel per margin (0% recovery)
  9_reliability.png        raw score vs calibrated default probability against actual bad rate
  1b_retrain_trigger.png   rolling 3-month AUC against the baseline and the amber and red limits (the alert series on its own)
  6_SIMULATED_stress_test.png  what a drift alert looks like (synthetic shift, NOT production data)
and datamart/gold/model_monitoring/monitoring_summary.csv
Every chart is also written at slide size to charts/deck/ (same data, same code; narrower canvas and larger relative type).
"""
import argparse
import glob
import pickle
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from utils.config import GOLD_DIR, MODEL_BANK_DIR
from utils.model_utils import load_features_month
from utils.psi import psi

p = argparse.ArgumentParser()
p.add_argument("--snapshotdate", required=True)
p.parse_args()

MON = GOLD_DIR / "model_monitoring"
OUT = MON / "charts"
DECK = OUT / "deck"
DECK.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(MON / "gold_model_monitoring_*.parquet")))
m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

BLUE, ORANGE, RED, INK, MUTED, GRID = "#0369A1", "#D97706", "#DC2626", "#1E293B", "#475569", "#E2E8F0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titlelocation": "left"})


def fig(w=10, h=4.6):
    f, ax = plt.subplots(figsize=(w, h))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return f, ax


def months(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())


def save(f, name, deck=None):
    f.tight_layout()
    f.savefig(OUT / name, dpi=200, facecolor="white")
    save_deck(f, name, **(deck or {}))
    plt.close(f)


def save_deck(f, name, width=6.2, height=None, ylim=None, xlabel=None, legend=None, title=True, xevery=None, drop_annotations=False, ylabel=None):
    """Same figure at slide size: narrower canvas so the type stays legible when placed in a slide."""
    w, h = f.get_size_inches()
    f.set_size_inches(width, height or h * width / w)
    for a in f.axes:
        for t in (a.title, a._left_title):
            t.set_fontsize(9)
            t.set_wrap(True)
        a.xaxis.label.set_fontsize(8)
        a.yaxis.label.set_fontsize(8)
        a.tick_params(labelsize=8)
        if width < 6:
            a.tick_params(axis="x", labelsize=7)
        for t in a.texts:
            t.set_fontsize(min(t.get_fontsize(), 7.5))
    ax = f.axes[0]
    if ylim:
        ax.set_ylim(*ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if not title:
        for a in f.axes:
            a.set_title(""); a.set_title("", loc="left")
    if xevery:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=xevery))
    if drop_annotations:
        for t in list(ax.texts):
            t.remove()
    if legend is False:
        for a in f.axes:
            if a.get_legend():
                a.get_legend().remove()
    elif legend is not None:
        hd, lb = ax.get_legend_handles_labels()
        ax.legend(hd, lb, frameon=False, fontsize=7, **legend)
    f.tight_layout()
    f.savefig(DECK / name, dpi=250, facecolor="white")


def series(metric, by="cohort_month"):
    return m[(m.metric_name == metric) & (m.segment == "all")].sort_values(by).set_index(by)


bundle_id = json.loads((MODEL_BANK_DIR / "champion.json").read_text())["model_id"]
with open(MODEL_BANK_DIR / f"{bundle_id}.pkl", "rb") as fh:
    bundle = pickle.load(fh)
base = bundle["baseline"]["auc"]

# 1) performance -------------------------------------------------------------
auc, roll = series("auc"), series("rolling_3m_auc")
f, ax = fig()
bl, bh = bundle["baseline"]["band_low"], bundle["baseline"]["band_high"]
ax.axhspan(bl, bh, color=GRID, alpha=0.6, lw=0, label=f"Reference band {bl:.3f}-{bh:.3f} (mean +/- 2 SD)")
ax.axhline(base, color=MUTED, lw=1.2, ls="--", label=f"Baseline: walk-forward AUC {base:.3f}")
bt = m[(m.metric_name == "auc") & (m.segment == "backtest_sealed")].sort_values("cohort_month").set_index("cohort_month")
if len(bt):
    ax.errorbar(bt.index, bt.metric_value, yerr=[bt.metric_value - bt.ci_low, bt.ci_high - bt.metric_value],
                fmt="D", mfc="white", color=MUTED, ecolor=MUTED, elinewidth=1.2, capsize=0, ms=7, mew=1.5,
                label="Backtest, loans after the training window (95% CI)")
ax.axhline(base - 0.03, color=ORANGE, lw=1.2, ls=":", label=f"Retrain trigger, amber: below {base - 0.03:.3f} (rolling, two runs)")
ax.axhline(base - 0.05, color=RED, lw=1.2, ls=":", label=f"Escalation, red: below {base - 0.05:.3f} (rolling, two runs)")
ax.errorbar(auc.index, auc.metric_value, yerr=[auc.metric_value - auc.ci_low, auc.ci_high - auc.metric_value],
            fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=0, ms=7, mec="white", mew=1.5, alpha=0.9,
            label="Monthly AUC (95% CI)")
ax.plot(roll.index, roll.metric_value, color=ORANGE, lw=2.2, label="Rolling 3-month AUC (alert series)")
ax.set_ylim(0.66, 0.90)
ax.set_ylabel("AUC")
ax.set_title("Monthly and rolling 3-month AUC by origination cohort")
ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
months(ax)
ax.set_xlabel("Origination cohort (label known 5 months later). Each rolling point pools that cohort and the two before it.")
save(f, "1_performance.png", deck=dict(width=5.6, height=3.3, ylim=(0.60, 0.90), xlabel="Origination cohort (label known 5 months later)", legend=dict(loc="lower left", ncol=1)))

# 1b) retrain-trigger view: the rolling AUC (the alert series) on its own, against the baseline and the two limits
f, ax = fig()
ax.axhline(base, color=MUTED, lw=1.3, ls="--")
ax.axhline(base - 0.03, color=ORANGE, lw=1.5, ls=":")
ax.axhline(base - 0.05, color=RED, lw=1.5, ls=":")
ax.plot(roll.index, roll.metric_value, color=BLUE, lw=2.6, marker="o", ms=6, mec="white", mew=1.5)
x0 = roll.index[0]
ax.text(x0, base - 0.004, f"Baseline {base:.3f}", color=MUTED, fontsize=9, va="top")
ax.text(x0, base - 0.03 + 0.003, f"Amber {base - 0.03:.3f}: retrain trigger", color=ORANGE, fontsize=9, va="bottom")
ax.text(x0, base - 0.05 - 0.003, f"Red {base - 0.05:.3f}: escalation", color=RED, fontsize=9, va="top")
ax.set_ylim(0.72, 0.86)
ax.set_ylabel("Rolling 3-month AUC")
ax.set_title("Rolling 3-month AUC against the retrain limits")
months(ax)
ax.set_xlabel("Origination cohort (label known 5 months later)")
save(f, "1b_retrain_trigger.png", deck=dict(width=2.9, height=2.5, xlabel="", ylabel="Rolling AUC", title=False, xevery=2, legend=False))

# 2) score + worst feature PSI ----------------------------------------------
sp = series("score_psi", "scoring_month")
fp = m[m.metric_name == "feature_psi"].groupby("scoring_month").metric_value.max()
f, ax = fig()
ax.axhline(0.10, color=ORANGE, lw=1.2, ls="--")
ax.text(sp.index[-1], 0.103, "Investigate > 0.10", color=ORANGE, fontsize=9, ha="right", va="bottom")
ax.axhline(0.25, color=RED, lw=1.2, ls="--")
ax.text(sp.index[-1], 0.253, "Retrain review > 0.25", color=RED, fontsize=9, ha="right", va="bottom")
ax.plot(sp.index, sp.metric_value, color=BLUE, lw=2.2, marker="o", ms=6, mec="white", mew=1.5, label="Score PSI")
ax.plot(fp.index, fp.values, color=MUTED, lw=1.8, marker="o", ms=5, mec="white", mew=1.2, label="Worst single-feature PSI")
ax.set_ylim(0, 0.30)
ax.set_ylabel("PSI vs training reference")
ax.set_title("Score PSI and worst single-feature PSI vs training reference")
ax.legend(loc="center right", frameon=False, fontsize=9)
months(ax)
ax.set_xlabel("Scoring month")
save(f, "2_stability_psi.png", deck=dict(width=2.9, height=1.3, xlabel="", ylabel="PSI", title=False, xevery=2, legend=False))

# 3) feature PSI heatmap -----------------------------------------------------
h = m[m.metric_name == "feature_psi"].pivot(index="segment", columns="scoring_month", values="metric_value")
h = h.loc[h.max(axis=1).sort_values(ascending=False).index]
f, ax = plt.subplots(figsize=(10, 7))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", BLUE])
im = ax.imshow(h.values, cmap=cmap, vmin=0, vmax=0.10, aspect="auto")
ax.set_xticks(range(h.shape[1]), [c.strftime("%b %y") for c in h.columns])
ax.set_yticks(range(h.shape[0]), h.index)
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        v = h.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.06 else INK)
ax.set_title("Feature PSI by month (colour scale caps at 0.10)", loc="left")
for s in ax.spines.values():
    s.set_visible(False)
save(f, "3_feature_psi_heatmap.png")

# 4) clickstream coverage ----------------------------------------------------
cv = series("clickstream_coverage", "scoring_month")
f, ax = fig()
ax.step(cv.index, cv.metric_value * 100, where="post", color=BLUE, lw=2.4)
ok, bad = cv[cv.metric_value >= 0.95], cv[cv.metric_value < 0.95]
ax.scatter(ok.index, ok.metric_value * 100, color=BLUE, s=55, zorder=3, edgecolor="white", linewidth=1.5, label="OK")
ax.scatter(bad.index, bad.metric_value * 100, color=RED, s=70, marker="s", zorder=3, edgecolor="white",
           linewidth=1.5, label="Red alert")
ax.legend(loc="center left", frameon=False, fontsize=9)
ax.axhline(95, color=RED, lw=1.2, ls=":")
ax.text(cv.index[-1], 89, "Red alert below 95% coverage", color=RED, fontsize=9, ha="right", va="top")
first_bad = cv[cv.metric_value < 0.95].index.min()
if pd.notna(first_bad):
    ax.annotate(f"RED ALERT {first_bad:%b %Y}:\nno clickstream rows for any\nnewly originated customer.\n"
                "Cause not established.\nModel unaffected (it does not\nuse clickstream); raised to\ndata owner",
                xy=(first_bad, 0), xytext=(first_bad + pd.Timedelta(days=20), 78), fontsize=9.5, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.2))
ax.set_ylim(-5, 108)
ax.set_ylabel("Loans with clickstream data (%)")
ax.set_title("Clickstream coverage of newly scored loans")
months(ax)
ax.set_xlabel("Scoring month")
save(f, "4_clickstream_coverage.png", deck=dict(width=2.9, height=1.3, xlabel="", ylabel="Coverage (%)", title=False, xevery=2, drop_annotations=True, legend=False))

# 5) calibration gap with 95% CI ---------------------------------------------
cg = series("calibration_gap")
f, ax = fig()
ax.axhspan(-3, 3, color=GRID, alpha=0.5, lw=0, label="Tolerance +/-3 pp")
ax.axhline(0, color=MUTED, lw=1)
ax.errorbar(cg.index, cg.metric_value * 100, yerr=[(cg.metric_value - cg.ci_low) * 100, (cg.ci_high - cg.metric_value) * 100],
            fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=0, ms=7, mec="white", mew=1.5,
            label="Mean default probability minus bad rate (95% CI)")
ax.set_ylim(-9, 9)
ax.set_ylabel("Percentage points")
ax.set_title("Calibration gap by cohort: mean default probability minus bad rate")
ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
months(ax)
ax.set_xlabel("Origination cohort")
save(f, "5_calibration.png", deck=dict(height=2.6, ylim=(-9, 12), xlabel="Origination cohort", legend=dict(loc="upper right", ncol=1)))

# 7) challenger vs champion on loans neither model has seen ----------------------
cl = m[m.metric_name == "challenger_auc_lift"].sort_values("scoring_month")
if len(cl):
    f, ax = fig()
    ax.axhspan(-0.03, 0.01, color=GRID, alpha=0.5, lw=0, label="Below promotion threshold (+0.010)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.errorbar(cl.cohort_month, cl.metric_value, yerr=[cl.metric_value - cl.ci_low, cl.ci_high - cl.metric_value],
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=0, ms=7, mec="white", mew=1.5,
                label="Challenger AUC minus champion AUC (95% CI), cumulative")
    ax.set_ylabel("AUC lift")
    ax.set_ylim(-0.02, 0.032)
    ax.set_title("Challenger vs champion on cumulative out-of-time cohorts")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    months(ax)
    ax.set_xlabel("Latest origination cohort included. Promotion needs lift >= +0.010, CI above 0 and human sign-off.")
    save(f, "7_challenger.png", deck=dict(width=2.9, height=2.2, xlabel="", title=False, legend=False))

# 8) cutoff trade-off and decision economics ----------------------------------
# Live-scored cohorts with known outcomes. Loss per bad loan is an exposure-based proxy (outstanding balance at
# instalment 5); the margin on a good loan and the recovery rate are ASSUMPTIONS (utils/config.py),
# shown as scenarios. Cutoffs are applied to the calibrated default probability.
from utils.config import MARGIN_SCENARIOS, RECOVERY_RATES, REFERENCE_CUTOFF, SILVER_DIR
pred_files = sorted(glob.glob(str(GOLD_DIR / "model_predictions" / "gold_model_predictions_*.parquet")))
econ = []
for pf in pred_files:
    pr = pd.read_parquet(pf)
    if "default_probability" not in pr:
        continue
    lm = pd.Timestamp(pr.snapshot_date.iloc[0]) + pd.DateOffset(months=5)
    tag = lm.strftime("%Y_%m_01")
    lp = GOLD_DIR / "label_store" / f"gold_label_store_{tag}.parquet"
    sp_ = SILVER_DIR / "loan_daily" / f"silver_loan_daily_{tag}.parquet"
    if not (lp.exists() and sp_.exists()):
        continue
    ead = pd.read_parquet(sp_, columns=["loan_id", "installment_num", "balance"])
    ead = ead[ead.installment_num == 5][["loan_id", "balance"]].rename(columns={"balance": "ead"})
    x = pr[["loan_id", "score", "default_probability"]].merge(pd.read_parquet(lp)[["loan_id", "label"]], on="loan_id").merge(ead, on="loan_id", how="left")
    x["cohort"] = pd.Timestamp(pr.snapshot_date.iloc[0])
    econ.append(x)
if econ:
    e = pd.concat(econ, ignore_index=True)
    n_all, n_bad, n_good = len(e), int(e.label.sum()), int((e.label == 0).sum())
    loss_mean = e[e.label == 1].ead.mean()
    cuts = np.round(np.arange(0.05, 0.9501, 0.05), 2)
    rows = []
    for c in cuts:
        appr = e.default_probability < c            # approved = probability below the cutoff; flagged = at or above
        r = {"cutoff": c, "flagged_share": 1 - appr.mean(),
             "bad_loans_caught": (~appr & (e.label == 1)).sum() / n_bad,
             "good_customers_flagged": (~appr & (e.label == 0)).sum() / n_good,
             "bad_rate_among_approved": e[appr].label.mean() if appr.any() else np.nan}
        bad_ead = e[appr & (e.label == 1)].ead.sum()
        for rec in RECOVERY_RATES:
            for name, mgn in MARGIN_SCENARIOS.items():
                r[f"net_value_per_1000_margin_{mgn}_recovery_{int(rec*100)}"] = ((appr & (e.label == 0)).sum() * mgn - bad_ead * (1 - rec)) / n_all * 1000
        rows.append(r)
    tr = pd.DataFrame(rows)
    be = []
    for rec in RECOVERY_RATES:
        for name, mgn in MARGIN_SCENARIOS.items():
            L = loss_mean * (1 - rec)
            be.append({"margin": mgn, "recovery": rec, "break_even_cutoff": mgn / (mgn + L)})
    be = pd.DataFrame(be)
    tr.round(4).to_csv(MON / "cutoff_tradeoff.csv", index=False)
    be.round(4).to_csv(MON / "cutoff_break_even.csv", index=False)
    f, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for a in (a1, a2):
        a.grid(axis="y", color=GRID, linewidth=0.8); a.set_axisbelow(True)
        a.axvline(REFERENCE_CUTOFF, color=MUTED, lw=1.2, ls="--")
    a1.plot(tr.cutoff, tr.bad_loans_caught * 100, color=BLUE, lw=2.2, marker="o", ms=5, label="Bad loans flagged")
    a1.plot(tr.cutoff, tr.good_customers_flagged * 100, color=ORANGE, lw=2.2, marker="o", ms=5, label="Good customers flagged")
    a1.set_xlabel("Cutoff on default probability (flag loans at or above)"); a1.set_ylabel("% of group")
    a1.set_title("Who gets flagged at each cutoff"); a1.legend(frameon=False, fontsize=9, loc="upper right")
    def value_panel(a2, legend_kw, title_size=11):
        cols_ = [BLUE, ORANGE, MUTED]
        for (name, mgn), col in zip(MARGIN_SCENARIOS.items(), cols_):
            for rec, ls in zip(RECOVERY_RATES, ["-", "--"]):
                v = tr[f"net_value_per_1000_margin_{mgn}_recovery_{int(rec*100)}"] / 1000
                a2.plot(tr.cutoff, v, color=col, lw=2.0, ls=ls, marker="o" if rec == 0 else None, ms=4,
                        label=f"Margin {name}, recovery {rec:.0%}")
                k = v.idxmax(); a2.scatter(tr.cutoff[k], v[k], s=100, facecolor="none", edgecolor=col, lw=2, zorder=4)
                bec = be[(be.margin == mgn) & (be.recovery == rec)].break_even_cutoff.iloc[0]
                if bec <= 0.95:
                    a2.scatter([bec], [0], marker="^", s=70, color=col, zorder=5, edgecolor="white")
        a2.axhline(0, color=INK, lw=1)
        # cutoff 0.95 flags (almost) nobody, so its value is the value of approving everyone; the gain at the
        # reference cutoff is the gap between each curve at 0.30 and at 0.95
        a2.axvline(tr.cutoff.max(), color=MUTED, lw=1, ls=":")
        a2.set_xlim(right=tr.cutoff.max() + 0.06)
        a2.text(tr.cutoff.max() + 0.01, a2.get_ylim()[1] * 0.98, "approve everyone", rotation=90, ha="left", va="top", fontsize=8, color=MUTED)
        a2.set_xlabel("Cutoff on default probability (approve below)")
        a2.set_ylabel("Net value per 1,000 applicants (thousands)")
        a2.set_title("Estimated value of approvals (circle = best, triangle = break-even)", fontsize=title_size)
        a2.legend(frameon=False, **legend_kw)

    value_panel(a2, dict(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2))
    f.text(0.01, 0.005, f"Live cohorts with known outcomes (n={n_all:,}; {n_bad/n_all:.0%} bad in this data). Dashed vertical = reference cutoff {REFERENCE_CUTOFF}.\n"
           f"Loss per bad loan = outstanding balance at instalment 5 (observed exposure, mean {loss_mean:,.0f}) less assumed recovery. Margins and recovery rates are assumptions, not data. Break-even = margin / (margin + loss).",
           fontsize=8, color=MUTED)
    f.tight_layout(rect=(0, 0.05, 1, 1)); f.savefig(OUT / "8_cutoff_tradeoff.png", dpi=200, facecolor="white"); plt.close(f)
    # slide-size version: the value panel on its own (the flagged shares go in the slide's table)
    fd, ad = plt.subplots(figsize=(6.2, 3.7))
    ad.grid(axis="y", color=GRID, linewidth=0.8); ad.set_axisbelow(True)
    ad.axvline(REFERENCE_CUTOFF, color=MUTED, lw=1.2, ls="--")
    value_panel(ad, dict(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2), title_size=9.5)
    ad.xaxis.label.set_fontsize(8); ad.yaxis.label.set_fontsize(8); ad.tick_params(labelsize=8)
    fd.tight_layout(); fd.savefig(DECK / "8_cutoff_tradeoff.png", dpi=250, facecolor="white"); plt.close(fd)

    # 8b) simple view: losses avoided against margin lost, one panel per margin (0% recovery); green band = net gain over
    # approving everyone. Same inputs as chart 8: calibrated default probability, outstanding balance at instalment 5.
    avoided = np.array([e[(e.default_probability >= c) & (e.label == 1)].ead.sum() for c in cuts]) / n_all * 1000 / 1e6
    good_flagged = np.array([((e.default_probability >= c) & (e.label == 0)).sum() for c in cuts])
    i0, sel = int(np.argmin(abs(cuts - REFERENCE_CUTOFF))), cuts <= 0.60
    GREEN = "#2e9e6b"

    def gain_chart(path, figsize, title, fs, dpi):
        fg, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
        for k, (a, (name, mgn)) in enumerate(zip(axs, MARGIN_SCENARIOS.items())):
            lost = good_flagged * mgn / n_all * 1000 / 1e6
            net = avoided - lost
            a.plot(cuts[sel], avoided[sel], color=BLUE, lw=2.4)
            a.plot(cuts[sel], lost[sel], color=ORANGE, lw=2.4)
            a.fill_between(cuts[sel], lost[sel], avoided[sel], color=GREEN, alpha=0.16, lw=0)
            a.axvline(REFERENCE_CUTOFF, color=MUTED, lw=1.1, ls="--")
            a.scatter([cuts[i0]] * 2, [avoided[i0], lost[i0]], color=[BLUE, ORANGE], s=34, zorder=4)
            a.set_ylim(0, 2.65); a.set_xlim(cuts[sel][0], cuts[sel][-1])
            a.set_title(f"Margin per good loan: {mgn:,}", fontsize=fs + 1)
            a.text(0.97, 0.96, f"At cutoff {REFERENCE_CUTOFF:.2f}: net gain {net[i0]:.2f}m", transform=a.transAxes, ha="right", va="top",
                   fontsize=fs, fontweight="bold", color=GREEN, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
            a.text(0.97, 0.865, f"Best cutoff tested: {cuts[int(np.argmax(net))]:.2f}", transform=a.transAxes, ha="right", va="top", fontsize=fs - 1, color=MUTED, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
            a.set_xlabel("Cutoff: decline at or above this default probability", fontsize=fs - 1)
            a.tick_params(labelsize=fs - 1)
            a.grid(axis="y", color=GRID, linewidth=0.8); a.set_axisbelow(True)
            if k == 0:
                a.set_ylabel("Millions per 1,000 applicants", fontsize=fs - 1)
                a.text(0.97, 0.70, "Losses avoided", transform=a.transAxes, ha="right", va="top", fontsize=fs, fontweight="bold", color=BLUE)
                a.text(0.97, 0.27, "Margin lost on\ngood customers", transform=a.transAxes, ha="right", va="top", fontsize=fs, fontweight="bold", color=ORANGE)
        if title:
            fg.suptitle(title, x=0.01, ha="left", fontsize=fs + 3, fontweight="bold")
        fg.text(0.01, 0.005, f"Live loans with known outcomes (n={n_all:,}; {n_bad/n_all:.0%} bad, unrealistically high). Green band = net gain over approving everyone.\n"
                f"Loss per bad loan observed (mean {loss_mean:,.0f}, 0% recovery); margins are assumptions, not data.", fontsize=fs - 2, color=MUTED)
        fg.tight_layout(rect=(0, 0.09, 1, 0.93 if title else 1))
        fg.savefig(path, dpi=dpi, facecolor="white"); plt.close(fg)

    gain_chart(OUT / "8b_cutoff_gain.png", (13, 4.6), "Declining risky loans avoids more in defaults than it costs in lost margin", 11, 200)
    gain_chart(DECK / "8b_cutoff_gain.png", (9.0, 2.15), None, 8, 250)
    print(f"[report] cutoff trade-off written ({n_all} scored loans with outcomes)")

    # 9) reliability: raw score vs calibrated default probability ---------------
    def groups(d, k):
        d = d.copy(); d["g"] = pd.qcut(d.score.rank(method="first"), k, labels=False)
        return d.groupby("g").agg(n=("label", "size"), raw=("score", "mean"), calibrated=("default_probability", "mean"), actual=("label", "mean"))
    def ece(g):
        w = g.n / g.n.sum()
        return (w * (g.raw - g.actual).abs()).sum() * 100, (w * (g.calibrated - g.actual).abs()).sum() * 100
    gp = groups(e, 10)
    last = e.cohort.max()
    gl = groups(e[e.cohort == last], 5)
    pd.concat([gp.assign(set="all live cohorts"), gl.assign(set=f"latest cohort {last:%Y-%m}")]).round(4).to_csv(MON / "reliability.csv")
    f, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, g, t in ((a1, gp, f"All live cohorts (n={n_all:,}), 10 equal groups"), (a2, gl, f"Latest cohort {last:%Y-%m} (n={int(gl.n.sum()):,}), 5 groups")):
        er, ec = ece(g)
        hi = max(g.raw.max(), g.calibrated.max(), g.actual.max()) * 105
        a.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="Perfect")
        a.plot(g.raw * 100, g.actual * 100, color=ORANGE, lw=2, marker="o", ms=6, label=f"Raw score (avg gap {er:.1f} pts)")
        a.plot(g.calibrated * 100, g.actual * 100, color=BLUE, lw=2, marker="o", ms=6, label=f"Default probability (avg gap {ec:.1f} pts)")
        a.set_xlabel("Predicted default rate (%)"); a.set_ylabel("Actual bad rate (%)"); a.set_title(t, fontsize=10)
        a.grid(color=GRID, linewidth=0.8); a.set_axisbelow(True); a.legend(frameon=False, fontsize=9, loc="upper left")
    f.text(0.01, 0.005, "Groups ordered by score. Calibration fitted on 2023 training data only; these loans were never used to fit it. Small groups are noisy (about 1.5 pts at ~450 loans per group).", fontsize=8, color=MUTED)
    f.tight_layout(rect=(0, 0.03, 1, 1)); f.savefig(OUT / "9_reliability.png", dpi=200, facecolor="white"); plt.close(f)
    # slide-size version: all live cohorts only
    fr, ar = plt.subplots(figsize=(2.9, 2.5))
    hi_ = max(gp.raw.max(), gp.calibrated.max(), gp.actual.max()) * 105
    ar.plot([0, hi_], [0, hi_], color=MUTED, lw=1, ls="--", label="Perfect")
    ar.plot(gp.raw * 100, gp.actual * 100, color=ORANGE, lw=2, marker="o", ms=4, label=f"Raw score (avg gap {ece(gp)[0]:.1f} pts)")
    ar.plot(gp.calibrated * 100, gp.actual * 100, color=BLUE, lw=2, marker="o", ms=4, label=f"Default probability (avg gap {ece(gp)[1]:.1f} pts)")
    ar.set_xlabel("Predicted default rate (%)", fontsize=8); ar.set_ylabel("Actual bad rate (%)", fontsize=8); ar.tick_params(labelsize=8)
    ar.grid(color=GRID, linewidth=0.8); ar.set_axisbelow(True)
    fr.tight_layout(); fr.savefig(DECK / "9_reliability.png", dpi=250, facecolor="white"); plt.close(fr)
    print(f"[report] reliability: all live avg gap raw {ece(gp)[0]:.2f} vs calibrated {ece(gp)[1]:.2f} pts; latest {ece(gl)[0]:.2f} vs {ece(gl)[1]:.2f}")

# 6) SIMULATED stress test ---------------------------------------------------
# Synthetic shift applied to the latest scored cohort's inputs to show what an alert looks like.
# Nothing here is written to the monitoring table.
cohort = series("score_psi", "scoring_month").index.max()
feat = load_features_month(cohort)
cols = bundle["features"]["numeric"] + bundle["features"]["categorical"]
shift = ["Outstanding_Debt", "Interest_Rate", "Num_of_Delayed_Payment", "Delay_from_due_date"]
steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
sim = []
for s in steps:
    x = feat.copy()
    for c in shift:
        x[c] = pd.to_numeric(x[c], errors="coerce") * (1 + s)
    sc = bundle["pipeline"].predict_proba(x[cols])[:, 1]
    sim.append((s, psi(bundle["psi_reference"]["score"], sc), sc.mean()))
sim = pd.DataFrame(sim, columns=["shift", "psi", "mean_score"])
f, ax = fig()
ax.axhline(0.10, color=ORANGE, lw=1.2, ls="--")
ax.axhline(0.25, color=RED, lw=1.2, ls="--")
ax.text(62, 0.103, "Investigate > 0.10", color=ORANGE, fontsize=9, va="bottom", ha="right")
ax.text(62, 0.253, "Retrain review > 0.25", color=RED, fontsize=9, va="bottom", ha="right")
ax.plot(sim["shift"] * 100, sim["psi"], color=BLUE, lw=2.2, marker="o", ms=6, mec="white", mew=1.5)
for r in sim.itertuples():
    ax.text(r.shift * 100 + 1.2, r.psi - 0.035, f"{r.psi:.2f}", ha="left", fontsize=8.5, color=INK)
ax.set_xlabel("Synthetic increase applied to debt, interest-rate and delinquency inputs (%)")
ax.set_ylabel("Score PSI")
ax.set_ylim(0, max(0.32, sim.psi.max() * 1.15))
ax.set_title("SIMULATED stress test: score PSI under a synthetic input shift")
ax.text(0.01, 0.97, "SIMULATED: synthetic shift on the %s cohort. Not production data.\nA real alert needs the breach in two consecutive months. Labels are unchanged, so AUC cannot be simulated." % f"{cohort:%b %Y}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=RED, fontweight="bold")
save(f, "6_SIMULATED_stress_test.png")
sim.to_csv(MON / "stress_test_simulated.csv", index=False)

# summary table ---------------------------------------------------------------
def col(metric, field="metric_value", by="scoring_month"):
    s = m[(m.metric_name == metric) & (m.segment == "all")]
    if metric == "challenger_auc_lift":          # latest challenger only
        s = m[m.metric_name == metric]
        s = s[s.model_id == s.sort_values("scoring_month").model_id.iloc[-1]] if len(s) else s
    return s.set_index(by)[field]


summ = pd.DataFrame({
    "cohort_evaluated": m[(m.metric_name == "auc") & (m.segment == "all")].set_index("scoring_month").cohort_month.dt.strftime("%Y-%m"),
    "backtest_cohort": m[(m.metric_name == "auc") & (m.segment == "backtest_sealed")].set_index("scoring_month").cohort_month.dt.strftime("%Y-%m"),
    "backtest_auc": m[(m.metric_name == "auc") & (m.segment == "backtest_sealed")].set_index("scoring_month").metric_value,
    "auc": col("auc"), "auc_ci_low": col("auc", "ci_low"), "auc_ci_high": col("auc", "ci_high"),
    "rolling_3m_auc": col("rolling_3m_auc"), "calibration_gap": col("calibration_gap"),
    "challenger_auc_lift": col("challenger_auc_lift"), "challenger_lift_ci_low": col("challenger_auc_lift", "ci_low"),
    "challenger_lift_ci_high": col("challenger_auc_lift", "ci_high"),
})
stab = pd.DataFrame({"score_psi": col("score_psi"),
                     "max_feature_psi": m[m.metric_name == "feature_psi"].groupby("scoring_month").metric_value.max(),
                     "clickstream_coverage": col("clickstream_coverage")})
alerts = (m[m.status.isin(["watch", "amber", "red"])].assign(a=lambda d: d.metric_name + "[" + d.segment + "]=" + d.status)
          .groupby("scoring_month").a.agg("; ".join).rename("alerts"))
summary = stab.join(summ, how="outer").join(alerts).reset_index().rename(columns={"index": "scoring_month"})
summary.round(4).to_csv(MON / "monitoring_summary.csv", index=False)
print(f"[report] wrote charts to {OUT} and summary ({len(summary)} rows)")
print(summary.round(3).to_string(index=False))
