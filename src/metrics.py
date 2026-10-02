"""Error metrics for medal-count predictions.  Self-test:  python src/metrics.py

All three follow the paper (Dobkowski, "2020 Summer Olympics Predictions Using
Machine Learning"):
  - rmse_all:        RMSE over every country (the paper's "average std dev")
  - rmse_top10:      RMSE over the 10 countries with the most ACTUAL medals
  - selection_score: MSE_all + 0.25 * MSE_top10, used to pick the final regressor
"""
import numpy as np
import pandas as pd


def rmse_all(y_true, y_pred):
    """Root mean squared error over all countries, in medals.

    RMSE = sqrt(mean((actual - predicted)^2)). Squaring punishes big misses much
    more than small ones (missing by 10 costs 100, missing by 1 costs 1), and the
    square root puts the result back in "medals", so 3.2 reads as "typically off
    by about 3 medals".
    """
    # np.asarray accepts lists, numpy arrays and pandas Series alike. We work on
    # plain arrays so pandas never tries to line rows up by index label.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_top10(df, y_pred):
    """RMSE over the 10 countries with the most ACTUAL medals in `df`.

    `df` must hold ONE Games year (e.g. all 2012 rows) and have a `medals` column.
    `y_pred` must be in the same row order as `df` (y_pred[i] is the prediction
    for row i).

    Why a separate top-10 score: ~58% of countries win 0 medals, so a model can get
    a low rmse_all just by predicting small numbers for everyone while badly
    missing the USA or China. This score looks only at the big medal winners.
    """
    # Top 10 only makes sense within one year. Mixing years would compare, say,
    # the USA in 2008 with the USA in 2012.
    assert df["year"].nunique() == 1, "rmse_top10 expects rows from a single year"

    actual = df["medals"].to_numpy()
    y_pred = np.asarray(y_pred, dtype=float)

    # Positions of the 10 largest actual medal counts. argsort(-actual) sorts
    # from most to fewest medals; kind="stable" makes ties at 10th place break
    # the same way every run (by row order), so results are reproducible.
    top = np.argsort(-actual, kind="stable")[:10]
    return rmse_all(actual[top], y_pred[top])


def selection_score(y_true, y_pred, df):
    """The paper's model-selection criterion: MSE_all + 0.25 * MSE_top10.

    Lower is better. It uses MSE (= RMSE squared) for both parts, like the paper.
    The 0.25 weight adds the big countries' error on top of the overall error,
    so the chosen model can't fit only the many small countries (low MSE_all,
    huge top-10 error) or only the few big ones.
    """
    mse_all = rmse_all(y_true, y_pred) ** 2
    mse_top10 = rmse_top10(df, y_pred) ** 2
    return mse_all + 0.25 * mse_top10


if __name__ == "__main__":
    # Tiny hand-checkable example: 12 countries with 1..12 medals.
    # Every prediction is perfect except the biggest country (12 medals),
    # which we predict as 6, an error of 6.
    df = pd.DataFrame({"year": 2012, "medals": np.arange(1, 13)})
    pred = df["medals"].to_numpy().copy()
    pred[-1] = 6

    # By hand:
    #   MSE_all   = 6^2 / 12 countries = 3.0  -> RMSE_all   = sqrt(3)   ~ 1.732
    #   MSE_top10 = 6^2 / 10 countries = 3.6  -> RMSE_top10 = sqrt(3.6) ~ 1.897
    #   selection = 3.0 + 0.25 * 3.6 = 3.9
    r_all = rmse_all(df["medals"], pred)
    r_top = rmse_top10(df, pred)
    score = selection_score(df["medals"], pred, df)
    print(f"rmse_all   = {r_all:.3f}  (expected 1.732)")
    print(f"rmse_top10 = {r_top:.3f}  (expected 1.897)")
    print(f"selection  = {score:.3f}  (expected 3.900)")
    assert np.isclose(r_all, np.sqrt(3.0))
    assert np.isclose(r_top, np.sqrt(3.6))
    assert np.isclose(score, 3.9)

    # Row order must not matter: shuffle rows and predictions TOGETHER and the
    # scores must stay identical (this catches index-alignment bugs).
    order = np.random.default_rng(42).permutation(len(df))
    df_shuf, pred_shuf = df.iloc[order].reset_index(drop=True), pred[order]
    assert np.isclose(rmse_top10(df_shuf, pred_shuf), r_top)
    assert np.isclose(selection_score(df_shuf["medals"], pred_shuf, df_shuf), score)

    # A perfect prediction scores exactly 0 everywhere.
    assert selection_score(df["medals"], df["medals"], df) == 0.0
    print("All metric self-tests passed.")
