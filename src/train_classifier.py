"""Stage 1: medal / no medal classifier.  Run:  python src/train_classifier.py

Train 1988-2008, validate on 2012. 2016 is NEVER touched here (Stage 2 final test).
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FEATURE_COLS, ROOT, TRAIN_YEARS, VAL_YEAR, load_data

SEED = 42
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)


def scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def build_models(cv):
    kw = dict(cv=cv, scoring="f1", n_jobs=-1)
    return {
        "LogisticRegression": GridSearchCV(
            scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
            {"model__C": [0.01, 0.1, 1, 10, 100]}, **kw),
        "SVC": GridSearchCV(
            scaled(SVC(probability=True, random_state=SEED)),
            {"model__C": [0.1, 1, 10, 100], "model__kernel": ["linear", "rbf"]}, **kw),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomizedSearchCV(
            RandomForestClassifier(random_state=SEED),
            {"n_estimators": [100, 200, 300, 500], "max_depth": [3, 5, 8, 12, None]},
            n_iter=10, random_state=SEED, **kw),
        "MLP": RandomizedSearchCV(
            scaled(MLPClassifier(max_iter=2000, random_state=SEED)),
            {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
             "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]},
            n_iter=8, random_state=SEED, **kw),
    }


def main():
    df = load_data()
    train = df[df["year"].between(*TRAIN_YEARS)]
    val = df[df["year"] == VAL_YEAR]
    assert 2016 not in set(train["year"]) | set(val["year"])
    X_tr, y_tr = train[FEATURE_COLS], train["won_medal"]
    X_va, y_va = val[FEATURE_COLS], val["won_medal"]
    print(f"train rows: {len(train)}  val rows (2012): {len(val)}")
    print(f"majority-class baseline on 2012: {max(y_va.mean(), 1 - y_va.mean()):.3f}\n")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fitted, rows, cms = {}, [], {}
    for name, est in build_models(cv).items():
        est.fit(X_tr, y_tr)
        is_search = isinstance(est, BaseSearchCV)
        best = est.best_estimator_ if is_search else est
        params = est.best_params_ if is_search else {}
        pred = best.predict(X_va)
        fitted[name] = best
        cms[name] = confusion_matrix(y_va, pred, labels=[0, 1])
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_va, pred),
            "precision": precision_score(y_va, pred, zero_division=0),
            "recall": recall_score(y_va, pred, zero_division=0),
            "f1": f1_score(y_va, pred, zero_division=0),
            "cv_f1": est.best_score_ if is_search else np.nan,
            "best_params": str(params),
        })
        print(f"{name:20s} acc={rows[-1]['accuracy']:.3f} f1={rows[-1]['f1']:.3f}  {params}")

    res = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False)
    res.to_csv(ROOT / "reports" / "classification_results.csv", index=False)
    print("\n", res.drop(columns="best_params").round(3).to_string(index=False))

    # confusion matrices
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.8))
    for ax, (name, cm) in zip(axes, cms.items()):
        ConfusionMatrixDisplay(cm, display_labels=["No medal", "Medal"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrices_2012.png", dpi=150); plt.close()

    # RandomForest feature importances
    imp = pd.Series(fitted["RandomForest"].feature_importances_, index=FEATURE_COLS).sort_values()
    plt.figure(figsize=(7, 5))
    imp.plot.barh()
    plt.title("RandomForest feature importances")
    plt.tight_layout(); plt.savefig(FIG / "rf_feature_importances.png", dpi=150); plt.close()

    # save best (by F1 on 2012, accuracy breaks ties)
    winner = res.iloc[0]["model"]
    joblib.dump(fitted[winner], ROOT / "models" / "classifier.joblib")
    (ROOT / "models" / "classifier_info.json").write_text(json.dumps(
        {"model": winner, "features": FEATURE_COLS, "trained_on": list(TRAIN_YEARS),
         "validated_on": VAL_YEAR}, indent=2))
    print(f"\nWINNER: {winner}  -> models/classifier.joblib")


if __name__ == "__main__":
    main()
