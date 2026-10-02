"""Stage 2: tune medal-count regressors.  Run:  python src/train_regressor.py

Train / tune on 1988-2008, compare models on 2012 (validation). 2016 is NEVER
touched here; it is the final test in evaluate.py.

Outputs:
  reports/regressor_results.csv        before/after-tuning scores on 2012
  reports/figures/regressor_tuning.png  before vs after bar chart
  models/regressor_template.joblib      the chosen tuned regressor (UNFITTED)
  reports/chosen_model.json             its name + hyperparameters
"""
import json
import os
import warnings

# scikit-learn 1.6.1 passes an option ("iprint") that newer SciPy no longer knows.
# The optimiser still runs normally; the warning is just noise, so hide it: in this
# process (filterwarnings) and in the parallel worker processes that n_jobs=-1
# starts (they read PYTHONWARNINGS when they launch).
warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
os.environ["PYTHONWARNINGS"] = "ignore:Unknown solver options"

import joblib
import matplotlib
matplotlib.use("Agg")                 # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, PoissonRegressor, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from features import FEATURE_COLS, ROOT, TRAIN_YEARS, VAL_YEAR, load_data
from metrics import rmse_all, rmse_top10, selection_score
from pipeline import TwoStageModel

SEED = 42
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def scaled(model):
    # StandardScaler rescales every feature to mean 0, std 1. Putting it INSIDE a
    # Pipeline means it is re-fitted on each training fold only, so the scaling
    # never peeks at the data it is evaluated on.
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def round_clip(pred):
    # Same post-processing as TwoStageModel.predict: whole medals, never negative.
    # Applying it to the baseline too keeps the comparison fair: the only
    # difference left between baseline and two-stage is the two-stage idea itself.
    return np.clip(np.round(pred), 0, None)


def build_models():
    """Default-parameter Stage 2 regressors ("before tuning").

    Why scale some and not others:
      - Linear / Ridge / Lasso / Poisson / SVR work with distances or with penalties
        on coefficient size. GDP is in the trillions while pct_athletes is about 0.1
        to 6, so unscaled, the big-number columns dominate (SVR's distances) or the
        penalty hits features unevenly (Ridge/Lasso shrink a coefficient more when
        its feature has small units). Scaling puts every feature on the same footing.
      - A random forest only asks yes/no questions like "is gdp > 4.1e11?". Scaling a
        column doesn't change which rows fall on each side of a split, so it has no
        effect and we skip it.
    max_iter for Lasso/Poisson is a "give the optimiser enough steps to finish"
    setting, not a hyperparameter, so it is the same before and after tuning.
    """
    return {
        "LinearRegression": scaled(LinearRegression()),
        "Ridge": scaled(Ridge(random_state=SEED)),
        "Lasso": scaled(Lasso(max_iter=10000, random_state=SEED)),
        "PoissonRegressor": scaled(PoissonRegressor(max_iter=1000)),
        "SVR": scaled(SVR()),
        "RandomForest": RandomForestRegressor(random_state=SEED),
    }


def build_searches(models, cv):
    """Hyperparameter searches. "model__alpha" means the `alpha` of the Pipeline
    step called "model" (that's how sklearn reaches inside a Pipeline)."""
    kw = dict(cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    # GridSearchCV tries EVERY combination; fine for small grids.
    # RandomizedSearchCV tries n_iter random combinations; used for the forest,
    # whose full grid (3*4*3*2 = 72 combos, each up to 500 trees) is slower.
    return {
        "Ridge": GridSearchCV(models["Ridge"],
                              {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]}, **kw),
        "Lasso": GridSearchCV(models["Lasso"],
                              {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}, **kw),
        "PoissonRegressor": GridSearchCV(models["PoissonRegressor"],
                                         {"model__alpha": [0.001, 0.01, 0.1, 1]}, **kw),
        # A LIST of grids: `degree` only means something for the poly kernel, so it
        # gets its own grid instead of pointlessly multiplying the others by 2.
        "SVR": GridSearchCV(models["SVR"], [
            {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1, 10, 100],
             "model__gamma": ["scale", "auto"], "model__epsilon": [0.1, 0.5, 1]},
            {"model__kernel": ["poly"], "model__degree": [2, 3], "model__C": [0.1, 1, 10, 100],
             "model__gamma": ["scale", "auto"], "model__epsilon": [0.1, 0.5, 1]},
        ], **kw),
        "RandomForest": RandomizedSearchCV(models["RandomForest"], {
            "n_estimators": [100, 200, 500], "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 1.0],
        }, n_iter=30, random_state=SEED, **kw),
    }


def two_stage_scores(clf, reg_template, train, val):
    """Fit Stage 2 on the training winners, keep the saved Stage 1 as-is, score 2012."""
    model = TwoStageModel(clf, reg_template).fit(train, refit_clf=False)
    pred = model.predict(val)
    return (rmse_all(val["medals"], pred), rmse_top10(val, pred),
            selection_score(val["medals"], pred, val))


def main():
    # ---- 1. data: split by Games year, never randomly ----
    df = load_data()
    train = df[df["year"].between(*TRAIN_YEARS)]          # 1988-2008
    val = df[df["year"] == VAL_YEAR]                      # 2012
    assert 2016 not in set(train["year"]) | set(val["year"])
    winners = train[train["medals"] > 0]                  # Stage 2's training rows
    print(f"train rows: {len(train)}  (medal winners: {len(winners)})   val rows (2012): {len(val)}")

    # Stage 1: the saved classifier, already trained on 1988-2008. Not retrained here.
    clf = joblib.load(ROOT / "models" / "classifier.joblib")

    # ---- 2. baseline: ONE linear regression on ALL rows (zeros included) ----
    # This is the "single regressor" approach the two-stage model must beat.
    base = LinearRegression().fit(train[FEATURE_COLS], train["medals"])
    base_raw = base.predict(val[FEATURE_COLS])
    base_pred = round_clip(base_raw)
    base_all, base_top = rmse_all(val["medals"], base_pred), rmse_top10(val, base_pred)
    print(f"\nBASELINE (LinearRegression on all rows) on 2012: "
          f"RMSE_all={base_all:.3f}  RMSE_top10={base_top:.3f}"
          f"   [unrounded: RMSE_all={rmse_all(val['medals'], base_raw):.3f}]")

    # ---- 3 + 4. Stage 2 models with DEFAULT params, inside the two-stage setup ----
    models = build_models()
    before = {}
    print("\nBEFORE TUNING (two-stage, 2012):")
    for name, reg in models.items():
        r_all, r_top, _ = two_stage_scores(clf, reg, train, val)
        before[name] = r_all
        print(f"  {name:18s} RMSE_all={r_all:.3f}  RMSE_top10={r_top:.3f}")

    # ---- 5. tuning: 5-fold cross-validation on the 1988-2008 medal winners ----
    # NOTE: shuffled K-fold mixes years, so a fold can train on 2008 and validate
    # on 1992, i.e. some "future" data leaks into training folds. That only affects
    # which hyperparameters get picked. The numbers we REPORT come from the
    # year-based splits (train <= 2008 -> 2012, train <= 2012 -> 2016), which never
    # let a model see the year it is scored on.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tuned = {"LinearRegression": models["LinearRegression"]}   # no hyperparameters
    best_params = {"LinearRegression": {}}
    print("\nTUNING (5-fold CV on training-year medal winners):")
    for name, search in build_searches(models, cv).items():
        search.fit(winners[FEATURE_COLS], winners["medals"])
        # best_params_ keys look like "model__alpha"; strip the prefix for display.
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        # clone(best_estimator_) = the winning settings, UNFITTED, ready to reuse.
        tuned[name] = clone(search.best_estimator_)
        cv_rmse = np.sqrt(-search.best_score_)   # scores are NEGATIVE MSE (higher = better)
        print(f"  {name:18s} CV RMSE={cv_rmse:6.3f}  best params: {best_params[name]}")

    # ---- 6. tuned models inside the two-stage setup, on 2012 ----
    rows = []
    print("\nAFTER TUNING (two-stage, 2012):")
    for name, reg in tuned.items():
        r_all, r_top, score = two_stage_scores(clf, reg, train, val)
        rows.append({"model": name, "rmse_all_before": before[name], "rmse_all_after": r_all,
                     "rmse_top10_after": r_top, "selection_score": score,
                     "best_params": json.dumps(best_params[name])})
        print(f"  {name:18s} RMSE_all={r_all:.3f}  RMSE_top10={r_top:.3f}  selection={score:.2f}")

    # ---- 7. results table ----
    res = pd.DataFrame(rows).sort_values("selection_score")
    res.to_csv(ROOT / "reports" / "regressor_results.csv", index=False)
    print("\n", res.drop(columns="best_params").round(3).to_string(index=False))

    # ---- 8. chart: before vs after RMSE_all, baseline as a dashed line ----
    # Fixed model order (as built) so bars don't move around between runs.
    order = list(models)
    plot = res.set_index("model").loc[order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bar_kw = dict(edgecolor="white", linewidth=1.5)   # thin white gap between the paired bars
    b1 = ax.bar(x - w / 2, plot["rmse_all_before"], w, label="Before tuning (defaults)",
                color="#2a78d6", **bar_kw)
    b2 = ax.bar(x + w / 2, plot["rmse_all_after"], w, label="After tuning", color="#eb6834", **bar_kw)
    ax.axhline(base_all, color="#52514e", linestyle="--", linewidth=1.5, zorder=2,
               label=f"Baseline: one LinearRegression, no classifier ({base_all:.2f})")
    for bars in (b1, b2):   # value labels on a white box so the dashed line can't cross them
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3, color="#52514e", zorder=3,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xticks(x, order)
    ax.set_ylabel("RMSE, medals (2012, all countries)")
    ax.set_title("Stage 2 regressors in the two-stage model: 2012 validation error, "
                 "before vs after tuning\n(lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(plot[["rmse_all_before", "rmse_all_after"]].max().max(), base_all) * 1.18)
    plt.tight_layout()
    plt.savefig(FIG / "regressor_tuning.png", dpi=150)
    plt.close()

    # ---- 9. pick the final regressor: lowest selection_score on 2012 ----
    winner = res.iloc[0]["model"]
    joblib.dump(tuned[winner], ROOT / "models" / "regressor_template.joblib")
    (ROOT / "reports" / "chosen_model.json").write_text(json.dumps({
        "model": winner, "params": best_params[winner],
        "selected_by": "lowest selection_score (MSE_all + 0.25*MSE_top10) on 2012",
        "tuned_on": list(TRAIN_YEARS), "validated_on": VAL_YEAR,
        "baseline_2012": {"rmse_all": base_all, "rmse_top10": base_top},
    }, indent=2))
    print(f"\nCHOSEN REGRESSOR: {winner} {best_params[winner]}  -> models/regressor_template.joblib")

    # ---- 10. models that got WORSE after tuning ----
    worse = res[res["rmse_all_after"] > res["rmse_all_before"]]
    if len(worse):
        print("\nGOT WORSE AFTER TUNING (RMSE_all on 2012):")
        for _, r in worse.iterrows():
            print(f"  {r['model']}: {r['rmse_all_before']:.3f} -> {r['rmse_all_after']:.3f}")
        print("  Why: tuning picks the settings with the lowest CROSS-VALIDATION error on the\n"
              "  1988-2008 winners. Those settings fit the tuning years best, not necessarily\n"
              "  2012 (overfitting to the tuning data). CV also mixes years and scores plain\n"
              "  regression error, while 2012 is scored after the classifier and rounding.")
    else:
        print("\nNo model got worse after tuning.")


if __name__ == "__main__":
    main()
