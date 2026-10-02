"""Final evaluation.  Run after train_regressor.py:  python src/evaluate.py

  2012 (validation): saved Stage 1 classifier (1988-2008) + chosen regressor fit
                     on 1988-2008 winners -> reports/predictions_2012.csv
  2016 (final test): BOTH stages retrained on 1988-2012 with the chosen settings,
                     then predict 2016 ONCE. Also a baseline LinearRegression.

Outputs (reports/): predictions_2012.csv, predictions_2016.csv, final_results.csv,
top5_2016.csv, figures/final_2016_scatter.png; models/final_two_stage.joblib
"""
import os
import warnings

# Hide the harmless sklearn-1.6.1 / new-SciPy "iprint" warning (see train_regressor.py).
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

from features import FEATURE_COLS, ROOT, TEST_YEAR, TRAIN_YEARS, VAL_YEAR, load_data
from metrics import rmse_all, rmse_top10
from pipeline import TwoStageModel

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
PAPER = {"baseline": 3.25, "two_stage": 2.26}   # paper's 2016 RMSE_all, for reference


def round_clip(pred):
    # Whole medals, never negative: the same post-processing TwoStageModel uses.
    return np.clip(np.round(pred), 0, None).astype(int)


def main():
    df = load_data()
    train = df[df["year"].between(*TRAIN_YEARS)]                 # 1988-2008
    val = df[df["year"] == VAL_YEAR]                             # 2012
    train_full = df[df["year"].between(TRAIN_YEARS[0], VAL_YEAR)]  # 1988-2012
    test = df[df["year"] == TEST_YEAR]                           # 2016

    # ---- 1. the two saved pieces ----
    clf = joblib.load(ROOT / "models" / "classifier.joblib")           # trained 1988-2008
    reg_template = joblib.load(ROOT / "models" / "regressor_template.joblib")  # unfitted
    print("Stage 1:", clf.steps[-1][1])
    print("Stage 2:", reg_template)

    # ---- 2. 2012 validation predictions (nothing here has seen 2012) ----
    val_model = TwoStageModel(clf, reg_template).fit(train, refit_clf=False)
    pred_2012 = val_model.predict(val)
    pd.DataFrame({
        "noc": val["noc"], "country": val["country"], "year": val["year"],
        "actual": val["medals"], "predicted": pred_2012,
        "p_medal": val_model.predict_proba_medal(val).round(3),   # Stage 1 probability, for the app
    }).to_csv(REPORTS / "predictions_2012.csv", index=False)
    print(f"\n2012 (validation): RMSE_all={rmse_all(val['medals'], pred_2012):.3f}  "
          f"RMSE_top10={rmse_top10(val, pred_2012):.3f}")

    # ---- 3. FINAL TEST on 2016 ----
    # Retrain BOTH stages on 1988-2012: same model types and hyperparameters as
    # chosen on 2012, but with one more Games of data. clone(clf) = the classifier's
    # settings without its 1988-2008 training, so it is re-learned on 1988-2012.
    final = TwoStageModel(clone(clf), reg_template).fit(train_full)
    pred_2s = final.predict(test)

    # Baseline: ONE LinearRegression on all 1988-2012 rows (zeros included).
    base = LinearRegression().fit(train_full[FEATURE_COLS], train_full["medals"])
    pred_base = round_clip(base.predict(test[FEATURE_COLS]))

    # ---- 4. headline numbers ----
    res = pd.DataFrame({
        "metric": ["rmse_all", "rmse_top10"],
        "baseline": [rmse_all(test["medals"], pred_base), rmse_top10(test, pred_base)],
        "two_stage": [rmse_all(test["medals"], pred_2s), rmse_top10(test, pred_2s)],
    })
    # % improvement = how much LOWER the two-stage error is than the baseline's.
    res["improvement_pct"] = (res["baseline"] - res["two_stage"]) / res["baseline"] * 100
    res.to_csv(REPORTS / "final_results.csv", index=False)
    print("\n2016 FINAL TEST (trained on 1988-2012):")
    print(res.round(3).to_string(index=False))
    for _, r in res.iterrows():   # say the direction in words, so a minus sign isn't misread
        word = "BETTER" if r["improvement_pct"] > 0 else "WORSE"
        print(f"  {r['metric']}: two-stage is {abs(r['improvement_pct']):.1f}% {word} than the baseline")
    print(f"(paper, 2016 RMSE_all: baseline {PAPER['baseline']}, final {PAPER['two_stage']})")

    # ---- 5. every country's 2016 prediction ----
    out = pd.DataFrame({
        "noc": test["noc"], "country": test["country"], "actual": test["medals"],
        "predicted_two_stage": pred_2s, "predicted_baseline": pred_base,
        "p_medal": final.predict_proba_medal(test).round(3),
    })
    out.to_csv(REPORTS / "predictions_2016.csv", index=False)

    # ---- 6. top 5 countries by actual 2016 medals ----
    top5 = out.nlargest(5, "actual")[["noc", "country", "actual", "predicted_two_stage",
                                      "predicted_baseline"]]
    top5.to_csv(REPORTS / "top5_2016.csv", index=False)
    print("\nTop 5 countries, 2016:")
    print(top5.to_string(index=False))

    # ---- 7. predicted vs actual scatter: baseline (left) vs two-stage (right) ----
    # Same axis limits on both panels so the two are directly comparable. Points on
    # the red y = x line are perfect predictions; below it = under-predicted.
    lim = max(out[["actual", "predicted_two_stage", "predicted_baseline"]].max()) * 1.08
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    panels = [("predicted_baseline", "Baseline: one LinearRegression", res.loc[0, "baseline"]),
              ("predicted_two_stage", f"Two-stage: classifier + {type(reg_template).__name__}",
               res.loc[0, "two_stage"])]
    for ax, (col, title, r) in zip(axes, panels):
        ax.plot([0, lim], [0, lim], color="#e34948", linewidth=1.5, label="perfect prediction (y = x)")
        ax.scatter(out["actual"], out[col], s=22, alpha=0.6, color="#2a78d6",
                   edgecolor="white", linewidth=0.5, label="one country")
        # Label only the 5 biggest countries, plus this panel's single largest miss
        # (the point the audience will ask about), to keep it readable.
        worst = out.loc[[(out[col] - out["actual"]).abs().idxmax()]]
        for _, row in pd.concat([top5, worst]).drop_duplicates("noc").iterrows():
            ax.annotate(row["noc"], (row["actual"], row[col]), xytext=(5, -10),
                        textcoords="offset points", fontsize=9, color="#0b0b0b")
        ax.set(xlim=(0, lim), ylim=(0, lim), title=f"{title}\nRMSE_all = {r:.2f} medals",
               xlabel="Actual medals, 2016")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted medals, 2016")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("2016 final test: predicted vs actual medals per country", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "final_2016_scatter.png", dpi=150)
    plt.close()

    # ---- 8. where does the two-stage error come from? ----
    # Stage 1 check: the classifier INSIDE the final model (retrained on 1988-2012).
    X = test[FEATURE_COLS]
    won_pred = final.clf_.predict(X)
    won_true = test["won_medal"].to_numpy()
    print("\nSTAGE 1 on 2016 (classifier retrained on 1988-2012):")
    print(f"  accuracy={accuracy_score(won_true, won_pred):.3f}  "
          f"precision={precision_score(won_true, won_pred):.3f}  "
          f"recall={recall_score(won_true, won_pred):.3f}  F1={f1_score(won_true, won_pred):.3f}")
    cm = confusion_matrix(won_true, won_pred, labels=[0, 1])
    print("  confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["actual: no medal", "actual: medal"],
                       columns=["pred: no medal", "pred: medal"]).to_string())

    # Split the SQUARED error (what RMSE is built from: RMSE^2 = total / n) by cause.
    actual = test["medals"].to_numpy()
    sq_err = (actual - pred_2s) ** 2
    missed = (won_true == 1) & (won_pred == 0)      # winner, but Stage 1 said 0
    false_alarm = (won_true == 0) & (won_pred == 1)  # no medal, but Stage 1 said yes
    both_yes = (won_true == 1) & (won_pred == 1)    # Stage 1 right; only the count can be off
    total = sq_err.sum()
    print("\n  Where the 2016 squared error comes from:")
    for name, mask in [("Stage 1 missed a medal winner (predicted 0)", missed),
                       ("Stage 1 false alarm (0-medal country predicted to win)", false_alarm),
                       ("Stage 2 count error (Stage 1 right, count off)", both_yes)]:
        print(f"    {name:55s} {mask.sum():3d} countries  {sq_err[mask].sum() / total * 100:5.1f}%")
    # What if Stage 1 were perfect? Use Stage 2's count exactly for the real winners.
    counts = round_clip(final.reg_.predict(X))
    oracle = np.where(won_true == 1, counts, 0)
    print(f"  RMSE_all with a PERFECT Stage 1: {rmse_all(actual, oracle):.3f} "
          f"(actual: {rmse_all(actual, pred_2s):.3f})")

    # ---- 9. save the final model for the demo app ----
    # compress=3 shrinks the 500-tree forest a lot. The pickle refers to the class
    # as pipeline.TwoStageModel, so whoever loads it must be able to `import pipeline`
    # (the app adds src/ to its import path for exactly this reason).
    joblib.dump(final, ROOT / "models" / "final_two_stage.joblib", compress=3)
    print("\nSaved models/final_two_stage.joblib (trained on 1988-2012)")


if __name__ == "__main__":
    main()
