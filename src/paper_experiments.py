"""SUPPLEMENTARY: experiments from the Dobkowski paper that our main pipeline skipped.
Run:  python src/paper_experiments.py      (after train_regressor.py and evaluate.py)

NOTHING here changes our chosen model (LogisticRegression + RandomForest), its settings,
or any existing output. Everything is written to NEW files:
  reports/paper_experiments.csv      one row per experiment/variant
  reports/paper_classifiers.csv      paper Table 3: the five classifiers on 2012
  reports/paper_top5_2016.csv        paper Table 4: GaussianNB + Ridge, top 5 in 2016
  reports/figures/paper_experiments.png

Same rules as the main pipeline, so every number is comparable:
  - tune on 1988-2008, score on 2012 (validation)
  - for test numbers retrain on 1988-2012 and score 2016 (and 2020 if
    data/processed/tokyo2020_features.csv exists: retrain on 1988-2016)
  - same metrics, same rounding to whole medals and clipping at 0
IMPORTANT: our model was chosen on 2012 BEFORE these extra candidates existed. If one
of them scores better, especially on 2016, we report it but do NOT switch: picking a
model because of its 2016 score would be choosing on the test set.
"""
import os
import warnings

# Same harmless sklearn-1.6.1 / new-SciPy warning as in train_regressor.py: hide it here
# and in the parallel worker processes.
warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
os.environ["PYTHONWARNINGS"] = "ignore:Unknown solver options"

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV

from evaluate import round_clip
from features import FEATURE_COLS, ROOT, TEST_YEAR, TRAIN_YEARS, VAL_YEAR, load_data
from metrics import rmse_all, rmse_top10, selection_score
from pipeline import TwoStageModel
from train_classifier import build_models as build_classifiers
from train_regressor import build_models as build_regressors
from train_regressor import build_searches, two_stage_scores

SEED = 42
REPORTS = ROOT / "reports"
TOKYO_CSV = ROOT / "data" / "processed" / "tokyo2020_features.csv"

# ---------------------------------------------------------------- data splits
df = load_data()
TRAIN = df[df["year"].between(*TRAIN_YEARS)]                    # 1988-2008: tune / fit
VAL = df[df["year"] == VAL_YEAR]                                # 2012: validation
FULL_2012 = df[df["year"].between(TRAIN_YEARS[0], VAL_YEAR)]    # 1988-2012: refit for 2016
TEST = df[df["year"] == TEST_YEAR]                              # 2016: test
# Tokyo 2020 only if someone has built that file (same columns as the main CSV).
TOKYO = pd.read_csv(TOKYO_CSV) if TOKYO_CSV.exists() else None
FULL_2016 = df[df["year"].between(TRAIN_YEARS[0], TEST_YEAR)]   # 1988-2016: refit for 2020
NO_2020 = "" if TOKYO is not None else "no 2020 data (data/processed/tokyo2020_features.csv missing)"

WINNERS = TRAIN[TRAIN["medals"] > 0]   # Stage 2 tunes on 1988-2008 medal winners only
KFOLD = KFold(n_splits=5, shuffle=True, random_state=SEED)   # same CV as train_regressor.py

# Our chosen model, loaded from its saved files (never re-chosen here).
CLF = joblib.load(ROOT / "models" / "classifier.joblib")              # LogisticRegression
REG = joblib.load(ROOT / "models" / "regressor_template.joblib")      # tuned RandomForest

ROWS = []   # one dict per experiment/variant -> reports/paper_experiments.csv


def scores(year_df, pred):
    """RMSE_all, RMSE_top10 and selection score for one year's predictions."""
    return (rmse_all(year_df["medals"], pred), rmse_top10(year_df, pred),
            selection_score(year_df["medals"], pred, year_df))


def add_row(experiment, variant, p2012, p2016, p2020=None, notes=""):
    """Score predictions for 2012 / 2016 (/ 2020) and store one results row."""
    a12, t12, s12 = scores(VAL, p2012)
    a16, t16, _ = scores(TEST, p2016)
    a20, t20 = (scores(TOKYO, p2020)[:2] if p2020 is not None else (np.nan, np.nan))
    ROWS.append({"experiment": experiment, "variant": variant,
                 "rmse_all_2012": a12, "rmse_top10_2012": t12, "selection_score_2012": s12,
                 "rmse_all_2016": a16, "rmse_top10_2016": t16,
                 "rmse_all_2020": a20, "rmse_top10_2020": t20,
                 "notes": "; ".join(n for n in [notes, NO_2020] if n)})
    print(f"  {variant:55s} 2012 RMSE={a12:6.3f} top10={t12:6.3f} sel={s12:7.2f} | "
          f"2016 RMSE={a16:6.3f} top10={t16:6.3f}")


def two_stage(experiment, variant, clf, reg, notes=""):
    """Score a two-stage model with the standard protocol.

    2012: both stages trained on 1988-2008.  2016: both retrained on 1988-2012.
    2020: both retrained on 1988-2016 (only if the Tokyo file exists).
    TwoStageModel clones clf and reg, so the templates passed in are never changed.
    (For our LogisticRegression, retraining on 1988-2008 reproduces the saved
    classifier exactly, so the 2012 numbers match train_regressor.py.)
    """
    p12 = TwoStageModel(clf, reg).fit(TRAIN).predict(VAL)
    p16 = TwoStageModel(clf, reg).fit(FULL_2012).predict(TEST)
    p20 = TwoStageModel(clf, reg).fit(FULL_2016).predict(TOKYO) if TOKYO is not None else None
    add_row(experiment, variant, p12, p16, p20, notes)


def baseline(experiment, variant, reg=None, notes=""):
    """Single regressor on ALL rows (zeros included), no classifier; rounded + clipped."""
    reg = LinearRegression() if reg is None else reg

    def fit_predict(train, test):
        return round_clip(clone(reg).fit(train[FEATURE_COLS], train["medals"]).predict(test[FEATURE_COLS]))

    p20 = fit_predict(FULL_2016, TOKYO) if TOKYO is not None else None
    add_row(experiment, variant, fit_predict(TRAIN, VAL), fit_predict(FULL_2012, TEST), p20, notes)


def best_of(search_or_model):
    """The tuned settings as an UNFITTED template (a plain model is returned as-is)."""
    if isinstance(search_or_model, BaseSearchCV):
        return clone(search_or_model.best_estimator_)
    return clone(search_or_model)


def scaled_gnb():
    """GaussianNB with standardised inputs.

    sklearn's GaussianNB adds var_smoothing * (the LARGEST feature variance) to EVERY
    feature's variance for numerical stability. Unscaled, the largest is GDP's (~8e23),
    so every feature gets ~8e14 added: features like pct_athletes (variance ~1) become
    flat and uninformative, and recall on 2012 drops to 0.57 (0.78 scaled). Scaling
    first gives every feature variance 1, so the smoothing is small for all of them.
    """
    return Pipeline([("scale", StandardScaler()), ("model", GaussianNB())])


def tuned_regressor(name):
    """A Stage 2 regressor with the hyperparameters train_regressor.py already chose
    (read from reports/regressor_results.csv), so we never re-tune what was tuned."""
    res = pd.read_csv(REPORTS / "regressor_results.csv").set_index("model")
    params = json.loads(res.loc[name, "best_params"])
    model = build_regressors()[name]
    prefix = "" if name == "RandomForest" else "model__"   # Pipelines name the step "model"
    return model.set_params(**{prefix + k: v for k, v in params.items()})


class TargetWeightedRegressor(BaseEstimator, RegressorMixin):
    """Paper Eq. 4: fit `base` with sample weights w_i = exp(-(y_i - y0)^2 / (2 tau^2)).

    y_i is each training country's medal count, so countries whose count is close to
    y0 get weight ~1 and the rest fade towards 0. tau sets how fast they fade: with
    tau=1, a country 3 medals away from y0 already gets weight exp(-4.5) ~ 0.01.
    Written as a tiny sklearn estimator so clone() and TwoStageModel work with it.
    """

    def __init__(self, base=None, y0=50.0, tau=1.0):
        self.base, self.y0, self.tau = base, y0, tau

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        w = np.exp(-(y - self.y0) ** 2 / (2 * self.tau ** 2))
        if w.max() == 0:   # every weight underflowed: nothing left to fit
            raise ValueError(f"all weights are 0 for y0={self.y0}, tau={self.tau}")
        # Effective sample size (Kish): how many equally weighted rows these weights
        # are "worth". Shows how much of the training data the fit really uses.
        self.n_eff_ = w.sum() ** 2 / (w ** 2).sum()
        self.est_ = clone(self.base)
        if isinstance(self.est_, Pipeline):   # weights go to the model step, not the scaler
            self.est_.fit(X, y, model__sample_weight=w)
        else:
            self.est_.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.est_.predict(X)


class ShareTargetRegressor(BaseEstimator, RegressorMixin):
    """Experiment 5: learn each country's SHARE of the medals, not its count.

    fit:     target = medals / total_medals_awarded (e.g. USA 2008: 110/957 = 0.115)
    predict: predicted share * that year's total_medals_awarded -> a medal count
    The total is one of the features, so it's available at prediction time.
    TwoStageModel then rounds and clips the count exactly as for every other model.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.est_ = clone(self.base).fit(X, np.asarray(y) / X["total_medals_awarded"].to_numpy())
        return self

    def predict(self, X):
        return self.est_.predict(X) * X["total_medals_awarded"].to_numpy()


# Experiment 3 feature maps. Plain module-level functions so the models can be
# cloned and pickled. Each returns a COPY with some columns replaced by their log.
def log_athletes(X):
    X = X.copy()
    X["athletes"] = np.log(X["athletes"])           # athletes >= 1, so log is defined
    return X


def log_athletes_gdp_pop(X):
    X = log_athletes(X)
    X["gdp"] = np.log(X["gdp"])                     # gdp and population are > 0
    X["population"] = np.log(X["population"])
    return X


def with_map(func, model):
    """Apply a feature map in front of a model (inside a Pipeline, so it's part of fit)."""
    return Pipeline([("logmap", FunctionTransformer(func)), ("model", model)])


# ---------------------------------------------------------------- reference rows
def reference_rows():
    print("\nREFERENCE: our baseline and our chosen model")
    baseline("reference", "Baseline: one LinearRegression, no classifier")
    two_stage("reference", "CHOSEN: LogisticRegression + RandomForest", CLF, REG,
              notes="our model, chosen on 2012 before these experiments")
    # Sanity check: these must equal the numbers the main pipeline reported.
    chosen = ROWS[-1]
    final = pd.read_csv(REPORTS / "final_results.csv").set_index("metric")
    assert np.isclose(chosen["rmse_all_2016"], final.loc["rmse_all", "two_stage"])
    assert np.isclose(ROWS[-2]["rmse_all_2016"], final.loc["rmse_all", "baseline"])


# ---------------------------------------------------------------- experiment 1
def exp1_classifiers():
    """Paper Table 3: tune all five classifiers exactly as train_classifier.py does
    (its build_models() grids and its StratifiedKFold), score them on 2012. Then, as
    an extra, put each one in front of OUR Stage 2 to see what it does to medal error.
    One difference: GaussianNB gets scaled inputs here (see scaled_gnb), so its row
    differs from reports/classification_results.csv, where it is unscaled."""
    print("\nEXPERIMENT 1: the five classifiers (paper Table 3)")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)   # same as train_classifier.py
    X, y = TRAIN[FEATURE_COLS], TRAIN["won_medal"]
    table, tuned = [], {}
    classifiers = build_classifiers(cv)
    classifiers["GaussianNB"] = scaled_gnb()   # train_classifier.py's version is unscaled
    for name, est in classifiers.items():
        est.fit(X, y)
        tuned[name] = best_of(est)
        model = est.best_estimator_ if isinstance(est, BaseSearchCV) else est
        pred = model.predict(VAL[FEATURE_COLS])
        table.append({"classifier": name,
                      "accuracy": accuracy_score(VAL["won_medal"], pred),
                      "precision": precision_score(VAL["won_medal"], pred, zero_division=0),
                      "recall": recall_score(VAL["won_medal"], pred, zero_division=0),
                      "f1": f1_score(VAL["won_medal"], pred, zero_division=0),
                      "best_params": str(est.best_params_) if isinstance(est, BaseSearchCV) else "{}"})
    table = pd.DataFrame(table).sort_values(["f1", "accuracy"], ascending=False)
    table.to_csv(REPORTS / "paper_classifiers.csv", index=False)
    print(table.drop(columns="best_params").round(3).to_string(index=False))

    # Same code, same data, same seeds -> must reproduce the main classifier results
    # (except GaussianNB, which is scaled here and unscaled there).
    main_res = pd.read_csv(REPORTS / "classification_results.csv").set_index("model")
    for _, r in table.iterrows():
        if r["classifier"] != "GaussianNB":
            assert np.isclose(r["f1"], main_res.loc[r["classifier"], "f1"]), r["classifier"]
    print("  (matches reports/classification_results.csv exactly, except the scaled GaussianNB)")

    # Extra: each OTHER classifier as Stage 1, with our tuned RandomForest as Stage 2.
    # (LogisticRegression + RandomForest is the CHOSEN reference row already.)
    for name, clf in tuned.items():
        if name != "LogisticRegression":
            two_stage("1: classifier swap", f"{name} + RandomForest", clf, REG,
                      notes="Stage 1 swapped, Stage 2 = our tuned RandomForest"
                            + ("; GaussianNB on scaled inputs" if name == "GaussianNB" else ""))
    return tuned


# ---------------------------------------------------------------- experiment 2
def exp2_lwlr():
    """Paper Eq. 4: locally weighted LinearRegression / Ridge / Lasso as Stage 2.
    (a) the paper's setting y0=50, tau=1; (b) a small grid, best picked on 2012."""
    print("\nEXPERIMENT 2: locally weighted regression (paper Eq. 4)")
    grid_rows = []
    for name in ["LinearRegression", "Ridge", "Lasso"]:
        base = tuned_regressor(name)   # Ridge/Lasso keep their tuned alpha
        n_eff = TargetWeightedRegressor(base, 50, 1).fit(
            WINNERS[FEATURE_COLS], WINNERS["medals"]).n_eff_
        two_stage("2: LWLR", f"{name}, paper setting (y0=50, tau=1)",
                  CLF, TargetWeightedRegressor(base, 50, 1),
                  notes=f"effective training rows {n_eff:.1f} of {len(WINNERS)}")

        # Grid: score every (y0, tau) on 2012 only; the best one also gets a 2016 score.
        for y0 in [10, 25, 50, 75]:
            for tau in [1, 5, 10, 25]:
                reg = TargetWeightedRegressor(base, y0, tau)
                try:
                    a, t, s = two_stage_scores(CLF, reg, TRAIN, VAL)
                    n_eff = clone(reg).fit(WINNERS[FEATURE_COLS], WINNERS["medals"]).n_eff_
                except ValueError as e:   # all weights underflowed to 0
                    a = t = s = n_eff = np.nan
                    print(f"  {name} y0={y0} tau={tau}: {e}")
                grid_rows.append({"model": name, "y0": y0, "tau": tau, "rmse_all_2012": a,
                                  "rmse_top10_2012": t, "selection_score_2012": s,
                                  "effective_rows": n_eff})
        grid = pd.DataFrame(grid_rows)
        best = grid[grid["model"] == name].sort_values("selection_score_2012").iloc[0]
        two_stage("2: LWLR", f"{name}, grid best on 2012 (y0={best.y0:.0f}, tau={best.tau:.0f})",
                  CLF, TargetWeightedRegressor(base, best.y0, best.tau),
                  notes=f"best of 16 (y0, tau) on 2012; effective rows {best.effective_rows:.0f}")
    pd.DataFrame(grid_rows).to_csv(REPORTS / "paper_lwlr_grid.csv", index=False)


# ---------------------------------------------------------------- experiment 3
def exp3_feature_maps():
    """Log feature maps for the baseline and the two-stage LinearRegression.
    log() turns multiplicative gaps (GDP 1000x bigger) into additive ones, which is
    what a linear model can represent. The paper reports this didn't help."""
    print("\nEXPERIMENT 3: log feature maps")
    lin2 = tuned_regressor("LinearRegression")   # scaled LinearRegression (Stage 2)
    two_stage("3: feature maps", "Two-stage LinearRegression, raw features", CLF, lin2,
              notes="comparison point for the log variants")
    for label, func in [("log(athletes)", log_athletes),
                        ("log(athletes, gdp, population)", log_athletes_gdp_pop)]:
        baseline("3: feature maps", f"Baseline + {label}",
                 reg=with_map(func, LinearRegression()))
        two_stage("3: feature maps", f"Two-stage LinearRegression + {label}", CLF,
                  with_map(func, clone(lin2)), notes="log applied to Stage 2 only; Stage 1 unchanged")


# ---------------------------------------------------------------- experiment 4
def exp4_sigmoid_svr():
    """SVR with a sigmoid kernel, tuned like the other SVR kernels (same C, gamma,
    epsilon values, same 5-fold CV on the 1988-2008 winners)."""
    print("\nEXPERIMENT 4: SVR with a sigmoid kernel")
    search = GridSearchCV(build_regressors()["SVR"], {
        "model__kernel": ["sigmoid"], "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto"], "model__epsilon": [0.1, 0.5, 1],
    }, cv=KFOLD, scoring="neg_mean_squared_error", n_jobs=-1)
    search.fit(WINNERS[FEATURE_COLS], WINNERS["medals"])
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"  best params: {params}  CV RMSE={np.sqrt(-search.best_score_):.3f}")
    two_stage("4: sigmoid SVR", "SVR sigmoid kernel (tuned)", CLF, best_of(search),
              notes=f"best params {params}")
    two_stage("4: sigmoid SVR", "SVR, our kernels (tuned best: linear)", CLF,
              tuned_regressor("SVR"), notes="from train_regressor.py, for comparison")


# ---------------------------------------------------------------- experiment 5
def exp5_share_target():
    """Stage 2 learns the SHARE of medals won, then multiplies back by the year's total.
    Tuned with the same searches as train_regressor.py, but on the share target."""
    print("\nEXPERIMENT 5: share-of-medals target")
    searches = build_searches(build_regressors(), KFOLD)
    share = WINNERS["medals"] / WINNERS["total_medals_awarded"]
    for name in ["RandomForest", "Ridge"]:
        search = searches[name].fit(WINNERS[FEATURE_COLS], share)
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        two_stage("5: share target", f"{name}, predicts share of medals (tuned)", CLF,
                  ShareTargetRegressor(best_of(search)), notes=f"best params {params}")
    two_stage("5: share target", "Ridge, predicts medal count (our tuned alpha)", CLF,
              tuned_regressor("Ridge"), notes="comparison point; RandomForest count = CHOSEN row")


# ---------------------------------------------------------------- experiment 6
def exp6_paper_pipeline():
    """The paper's final pipeline: GaussianNB classifier + Ridge regressor, with Ridge's
    alpha tuned on 1988-2008 like ours (it IS our tuned Ridge). Plus paper Table 4.
    GaussianNB gets scaled inputs (see scaled_gnb)."""
    print("\nEXPERIMENT 6: the paper's final pipeline (GaussianNB + Ridge)")
    gnb = scaled_gnb()
    ridge = tuned_regressor("Ridge")
    two_stage("6: paper pipeline", "PAPER: GaussianNB + Ridge", gnb, ridge,
              notes=f"GaussianNB on scaled inputs; Ridge alpha="
                    f"{ridge.get_params()['model__alpha']} (tuned on 1988-2008)")

    # Paper Table 4: top 5 countries in 2016, actual vs predicted (trained on 1988-2012).
    paper = TwoStageModel(gnb, ridge).fit(FULL_2012).predict(TEST)
    ours = TwoStageModel(CLF, REG).fit(FULL_2012).predict(TEST)
    table = pd.DataFrame({"noc": TEST["noc"], "country": TEST["country"], "actual": TEST["medals"],
                          "paper_gnb_ridge": paper, "ours_logreg_rf": ours}).nlargest(5, "actual")
    table.to_csv(REPORTS / "paper_top5_2016.csv", index=False)
    print("  Paper Table 4 (top 5, 2016):")
    print(table.to_string(index=False))


def make_chart(res):
    """RMSE_all on 2012 and 2016 for every row, grouped horizontal bars.
    Baseline and chosen model are highlighted with a shaded band and bold labels.
    A few variants are 4-10x worse than the rest, so the axis is capped and their
    true value is written next to the clipped bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = 6.0
    labels = [f"[{e.split(':')[0]}] {v}" if e != "reference" else v
              for e, v in zip(res["experiment"], res["variant"])]
    y = np.arange(len(res))[::-1]   # first row at the top
    h = 0.38
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(res) + 1.6))
    highlight = res["experiment"].eq("reference").to_numpy()
    for yi, hi in zip(y, highlight):
        if hi:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#ecebe6", zorder=0)
    for col, off, color, name in [("rmse_all_2012", h / 2, "#2a78d6", "2012 (validation)"),
                                  ("rmse_all_2016", -h / 2, "#eb6834", "2016 (test)")]:
        vals = res[col].to_numpy()
        ax.barh(y + off, np.minimum(vals, cap), h, color=color, label=name,
                edgecolor="white", linewidth=1, zorder=2)
        for yi, v in zip(y, vals):
            txt = f"{v:.2f}" + ("  (axis capped)" if v > cap else "")
            ax.text(min(v, cap) + 0.05, yi + off, txt, va="center", fontsize=7.5,
                    color="#52514e", zorder=3)
    ax.set_yticks(y, labels, fontsize=8.5)
    for tick, hi in zip(ax.get_yticklabels(), highlight):
        tick.set_fontweight("bold" if hi else "normal")
    ax.set_xlim(0, cap + 1.2)
    ax.set_xlabel("RMSE, medals (all countries; lower is better)")
    ax.set_title("Paper experiments: every variant's error on 2012 (validation) and 2016 (test)\n"
                 "shaded = our baseline and our chosen model", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, zorder=1)
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS / "figures" / "paper_experiments.png", dpi=150)
    plt.close()


def main():
    reference_rows()
    exp1_classifiers()
    exp2_lwlr()
    exp3_feature_maps()
    exp4_sigmoid_svr()
    exp5_share_target()
    exp6_paper_pipeline()

    res = pd.DataFrame(ROWS)
    res.to_csv(REPORTS / "paper_experiments.csv", index=False)
    print(f"\nSaved {len(res)} rows -> reports/paper_experiments.csv")
    make_chart(res)
    print("Saved reports/figures/paper_experiments.png")

    # Close the parallel worker processes (n_jobs=-1) before Python exits. Without
    # this, Windows sometimes prints a harmless but alarming "PermissionError" from
    # workers being torn down after all the work is already saved.
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)


if __name__ == "__main__":
    main()
