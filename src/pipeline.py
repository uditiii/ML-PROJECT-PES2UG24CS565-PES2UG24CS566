"""Two-stage medal model.  Self-test:  python src/pipeline.py

Stage 1 (classifier): will the country win ANY medal?          -> 0 or 1
Stage 2 (regressor):  if yes, HOW MANY? Trained only on winners -> a count
Final prediction = 0 if Stage 1 says no medal, else Stage 2's count.

Why two stages: ~58% of countries win 0 medals. A single regressor trained on all
countries is pulled toward small numbers by that pile of zeros, so it under-predicts
the big winners and still predicts fractional medals for countries that win none.
Splitting the question ("any?" then "how many?") lets each model do one job.
"""
import numpy as np
from sklearn.base import clone

from features import FEATURE_COLS


class TwoStageModel:
    def __init__(self, clf, reg):
        # clf and reg are TEMPLATES: sklearn estimators (or Pipelines) that say
        # WHICH model and WHICH hyperparameters to use. fit() trains copies of them.
        self.clf = clf
        self.reg = reg

    def fit(self, df, refit_clf=True):
        """Train both stages on the rows in `df` (e.g. all 1988-2008 rows).

        refit_clf=False means `clf` is ALREADY trained (e.g. the saved Stage 1
        classifier, trained on 1988-2008) and is used exactly as-is; only Stage 2
        is trained here.
        """
        X = df[FEATURE_COLS]

        # clone() makes a fresh, UNFITTED copy with the same settings. Without it,
        # .fit() would retrain the object the caller passed in. That matters here
        # because we reuse the same template many times (for 2012, then again for
        # 2016) and because the saved classifier is already trained. Fitting it
        # in place would silently overwrite the 1988-2008 version with a new one.
        self.reg_ = clone(self.reg)
        if refit_clf:
            # Stage 1 learns from ALL countries: winners (1) and non-winners (0).
            self.clf_ = clone(self.clf)
            self.clf_.fit(X, df["won_medal"])
        else:
            self.clf_ = self.clf   # already trained: use it untouched

        # Stage 2 learns ONLY from countries that won at least one medal, so the
        # zeros don't drag its predictions down. It answers "given that you win
        # something, how much?"
        winners = df["medals"] > 0
        self.reg_.fit(X[winners], df.loc[winners, "medals"])
        return self

    def predict(self, df):
        """Final medal predictions (whole numbers >= 0) for every row in `df`."""
        X = df[FEATURE_COLS]
        won = self.clf_.predict(X)              # 1 = will win a medal, 0 = won't
        counts = self.reg_.predict(X)           # Stage 2 runs on every row...

        # Medals are whole numbers and can't be negative. Linear models can output
        # e.g. -1.7 for a tiny country or 12.4 for a mid-sized one, so round to
        # the nearest integer and clip anything below 0 up to 0.
        counts = np.clip(np.round(counts), 0, None)

        # ...but its count is only used where Stage 1 said "medal"; elsewhere 0.
        return np.where(won == 1, counts, 0).astype(int)

    def predict_proba_medal(self, df):
        """Stage 1's probability (0..1) that each row's country wins a medal."""
        # predict_proba returns one column per class [P(0), P(1)]; take P(1).
        return self.clf_.predict_proba(df[FEATURE_COLS])[:, 1]


if __name__ == "__main__":
    # Smoke test on the real data with simple default models: train on
    # 1988-2008, predict 2012, and check the basic promises of the class.
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from features import load_data
    from metrics import rmse_all

    df = load_data()
    train, val = df[df["year"].between(1988, 2008)], df[df["year"] == 2012]

    clf_template = Pipeline([("scale", StandardScaler()),
                             ("model", LogisticRegression(max_iter=2000, random_state=42))])
    model = TwoStageModel(clf_template, LinearRegression()).fit(train)
    pred = model.predict(val)

    print("first 5 predictions for 2012:", dict(zip(val["noc"].head(), pred[:5].tolist())))
    print(f"RMSE_all on 2012: {rmse_all(val['medals'], pred):.2f}")
    assert pred.dtype.kind == "i" and (pred >= 0).all()          # ints, never negative
    assert ((pred > 0) <= (model.clf_.predict(val[FEATURE_COLS]) == 1)).all()  # 0 if Stage 1 says no
    proba = model.predict_proba_medal(val)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert not hasattr(clf_template, "classes_")   # clone(): the template was NOT fitted
    print("TwoStageModel self-test passed.")
