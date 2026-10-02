"""Tokyo 2020 as a SECOND final test year.  Run after prep_2020.py:  python src/predict_2020.py

The model is already chosen (on 2012) and tested (on 2016). Here we retrain the SAME
two-stage model (same classifier and regressor settings) and the SAME baseline on all
of 1988-2016, then predict 2020 once. Nothing is changed based on what 2020 shows.

Outputs (reports/): predictions_2020.csv, final_results_2020.csv, top10_2020.csv,
figures/tokyo2020_scatter.png
"""
import os
import warnings

# Same harmless sklearn-1.6.1 / new-SciPy "iprint" warning as elsewhere.
warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
os.environ["PYTHONWARNINGS"] = "ignore:Unknown solver options"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from evaluate import round_clip
from features import FEATURE_COLS, ROOT, load_data
from metrics import rmse_all, rmse_top10
from pipeline import TwoStageModel

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"


def main():
    train = load_data()   # every Games we have: 1988-2016
    test = pd.read_csv(ROOT / "data" / "processed" / "tokyo2020_features.csv")
    print(f"train: {train['year'].min()}-{train['year'].max()}, {len(train)} rows | "
          f"test: Tokyo 2020, {len(test)} countries")

    # ---- the SAME model and baseline, retrained on one more Games (1988-2016) ----
    clf = joblib.load(ROOT / "models" / "classifier.joblib")                 # settings only
    reg_template = joblib.load(ROOT / "models" / "regressor_template.joblib")
    model = TwoStageModel(clone(clf), reg_template).fit(train)
    pred_2s = model.predict(test)
    base = LinearRegression().fit(train[FEATURE_COLS], train["medals"])
    pred_base = round_clip(base.predict(test[FEATURE_COLS]))

    # ---- headline numbers (same layout as final_results.csv) ----
    res = pd.DataFrame({
        "metric": ["rmse_all", "rmse_top10"],
        "baseline": [rmse_all(test["medals"], pred_base), rmse_top10(test, pred_base)],
        "two_stage": [rmse_all(test["medals"], pred_2s), rmse_top10(test, pred_2s)],
    })
    res["improvement_pct"] = (res["baseline"] - res["two_stage"]) / res["baseline"] * 100
    res.to_csv(REPORTS / "final_results_2020.csv", index=False)
    print("\nTOKYO 2020 TEST (trained on 1988-2016):")
    print(res.round(3).to_string(index=False))
    for _, r in res.iterrows():
        word = "BETTER" if r["improvement_pct"] > 0 else "WORSE"
        print(f"  {r['metric']}: two-stage is {abs(r['improvement_pct']):.1f}% {word} than the baseline")

    # ---- every country's prediction; top 10 table ----
    out = pd.DataFrame({
        "noc": test["noc"], "country": test["country"], "actual": test["medals"],
        "predicted_two_stage": pred_2s, "predicted_baseline": pred_base,
        "p_medal": model.predict_proba_medal(test).round(3),
    })
    out.to_csv(REPORTS / "predictions_2020.csv", index=False)
    top10 = out.nlargest(10, "actual")[["noc", "country", "actual", "predicted_two_stage",
                                        "predicted_baseline"]]
    top10.to_csv(REPORTS / "top10_2020.csv", index=False)
    print("\nTop 10 countries, 2020:")
    print(top10.to_string(index=False))

    # ---- biggest misses of the two-stage model ----
    out["error"] = out["predicted_two_stage"] - out["actual"]
    worst = out.reindex(out["error"].abs().sort_values(ascending=False).index).head(3)
    feats = test.set_index("noc")
    print("\nBiggest two-stage misses (+ = over-predicted):")
    for _, r in worst.iterrows():
        f = feats.loc[r["noc"]]
        print(f"  {r['country']:12s} actual {r['actual']:3d}  predicted {r['predicted_two_stage']:3d} "
              f"({r['error']:+d})  athletes {f['athletes']}, medals in 2016 {f['medals_last_games']}")

    # ---- predicted vs actual scatter, styled like the 2016 one ----
    lim = max(out[["actual", "predicted_two_stage", "predicted_baseline"]].max()) * 1.08
    top5 = out.nlargest(5, "actual")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    panels = [("predicted_baseline", "Baseline: one LinearRegression", res.loc[0, "baseline"]),
              ("predicted_two_stage", f"Two-stage: classifier + {type(reg_template).__name__}",
               res.loc[0, "two_stage"])]
    for ax, (col, title, r) in zip(axes, panels):
        ax.plot([0, lim], [0, lim], color="#e34948", linewidth=1.5, label="perfect prediction (y = x)")
        ax.scatter(out["actual"], out[col], s=22, alpha=0.6, color="#2a78d6",
                   edgecolor="white", linewidth=0.5, label="one country")
        # Label the 5 biggest countries plus this panel's 3 largest misses. Close
        # neighbours (GBR 65 and RUS 71) would collide, so labels are staggered at two
        # depths (in order of actual medals) and each gets a thin leader line to its
        # own point, so no label can be read as belonging to the wrong dot.
        misses = out.loc[(out[col] - out["actual"]).abs().nlargest(3).index]
        labelled = pd.concat([top5, misses]).drop_duplicates("noc").sort_values("actual")
        for i, (_, row) in enumerate(labelled.iterrows()):
            ax.annotate(row["noc"], (row["actual"], row[col]),
                        xytext=(10, -16 - 14 * (i % 2)), textcoords="offset points",
                        fontsize=9, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6))
        ax.set(xlim=(0, lim), ylim=(0, lim), title=f"{title}\nRMSE_all = {r:.2f} medals",
               xlabel="Actual medals, Tokyo 2020")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted medals, Tokyo 2020")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Tokyo 2020 (second test year): predicted vs actual medals per country", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "tokyo2020_scatter.png", dpi=150)
    plt.close()

    # ---- Stage 1 check and error breakdown (same as evaluate.py) ----
    X = test[FEATURE_COLS]
    won_pred, won_true = model.clf_.predict(X), test["won_medal"].to_numpy()
    print("\nSTAGE 1 on 2020 (classifier retrained on 1988-2016):")
    print(f"  accuracy={accuracy_score(won_true, won_pred):.3f}  "
          f"precision={precision_score(won_true, won_pred):.3f}  "
          f"recall={recall_score(won_true, won_pred):.3f}  F1={f1_score(won_true, won_pred):.3f}")
    cm = confusion_matrix(won_true, won_pred, labels=[0, 1])
    print(pd.DataFrame(cm, index=["actual: no medal", "actual: medal"],
                       columns=["pred: no medal", "pred: medal"]).to_string())
    actual = test["medals"].to_numpy()
    sq_err = (actual - pred_2s) ** 2
    total = sq_err.sum()
    print("\n  Where the 2020 squared error comes from:")
    for name, mask in [("Stage 1 missed a medal winner (predicted 0)", (won_true == 1) & (won_pred == 0)),
                       ("Stage 1 false alarm (0-medal country predicted to win)", (won_true == 0) & (won_pred == 1)),
                       ("Stage 2 count error (Stage 1 right, count off)", (won_true == 1) & (won_pred == 1))]:
        print(f"    {name:55s} {mask.sum():3d} countries  {sq_err[mask].sum() / total * 100:5.1f}%")
    oracle = np.where(won_true == 1, round_clip(model.reg_.predict(X)), 0)
    print(f"  RMSE_all with a PERFECT Stage 1: {rmse_all(actual, oracle):.3f} "
          f"(actual: {rmse_all(actual, pred_2s):.3f})")


if __name__ == "__main__":
    main()
