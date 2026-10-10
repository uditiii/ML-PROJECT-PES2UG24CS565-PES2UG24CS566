"""SUPPLEMENTARY: compare under the PAPER'S protocol.  Run:  python src/paper_protocol.py

Why: the paper's Table 4 "2016 actual" medals equal the 2012 Games (USA 103, UK 65,
GER 44, JPN 38; real 2016: 121, 67, 42, 41), and the paper says its final model was
trained on 1988-2012. So its reported numbers look like 2012 predictions from a model
that had already SEEN 2012, i.e. IN-SAMPLE: scored on its own training data.

This script reproduces that protocol for comparison only, next to our honest
out-of-sample numbers. Nothing here changes our model, its settings or any existing
result file. Outputs (new files):
  reports/paper_protocol.csv          RMSE per model x protocol
  reports/paper_protocol_table4.csv   the paper's Table 4 countries, side by side
"""
import os
import warnings

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
os.environ["PYTHONWARNINGS"] = "ignore:Unknown solver options"

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from evaluate import round_clip
from features import FEATURE_COLS, ROOT, load_data
from metrics import rmse_all, rmse_top10
from paper_experiments import scaled_gnb, tuned_regressor
from pipeline import TwoStageModel

REPORTS = ROOT / "reports"

# The paper's Table 4: its predictions for these five countries. The paper states no MAE;
# 4.8 is what its predictions score against the actuals printed in the same table
# (USA 103, China 89, UK 65, Germany 44, Japan 38).
PAPER_TABLE4 = {"USA": 103, "CHN": 85, "GBR": 52, "GER": 44, "JPN": 45}
PAPER_TABLE4_MAE = 4.8


class Baseline:
    """Our baseline (one LinearRegression on all rows, rounded and clipped), wrapped
    with the same fit/predict interface as TwoStageModel so both can be looped over."""

    def fit(self, df):
        self.reg_ = LinearRegression().fit(df[FEATURE_COLS], df["medals"])
        return self

    def predict(self, df):
        return round_clip(self.reg_.predict(df[FEATURE_COLS]))


def main():
    df = load_data()
    tokyo = pd.read_csv(ROOT / "data" / "processed" / "tokyo2020_features.csv")
    games = {2012: df[df["year"] == 2012], 2016: df[df["year"] == 2016], 2020: tokyo}

    clf = joblib.load(ROOT / "models" / "classifier.joblib")
    reg = joblib.load(ROOT / "models" / "regressor_template.joblib")
    # Fresh model objects every time, so no fit ever leaks into another.
    models = {
        "Ours: LogisticRegression + RandomForest": lambda: TwoStageModel(clone(clf), reg),
        # GaussianNB on scaled inputs: see paper_experiments.scaled_gnb for why.
        "Paper's pipeline: GaussianNB + Ridge": lambda: TwoStageModel(scaled_gnb(),
                                                                      tuned_regressor("Ridge")),
        "Baseline: one LinearRegression": Baseline,
    }

    # (protocol label, training years, test year, in-sample?)
    # The paper's protocol trains on 1988-2012 and then scores 2012 itself: IN-SAMPLE.
    # Ours always scores a Games the model has never seen: OUT-OF-SAMPLE.
    protocols = [
        ("PAPER PROTOCOL: train 1988-2012, score 2012", (1988, 2012), 2012, True),
        ("Ours: train 1988-2008, score 2012 (validation)", (1988, 2008), 2012, False),
        ("Ours: train 1988-2012, score 2016 (test)", (1988, 2012), 2016, False),
        ("Ours: train 1988-2016, score 2020 (second test)", (1988, 2016), 2020, False),
    ]

    rows, preds_2012 = [], {}
    for name, make in models.items():
        for label, (lo, hi), year, in_sample in protocols:
            model = make().fit(df[df["year"].between(lo, hi)])
            test = games[year]
            pred = model.predict(test)
            rows.append({"model": name, "protocol": label,
                         "sample": "IN-SAMPLE (scored on training data)" if in_sample
                         else "out-of-sample", "trained_on": f"{lo}-{hi}", "scored_on": year,
                         "rmse_all": rmse_all(test["medals"], pred),
                         "rmse_top10": rmse_top10(test, pred)})
            if year == 2012:
                preds_2012[(name, in_sample)] = pd.Series(pred, index=test["noc"].to_numpy())

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "paper_protocol.csv", index=False)
    print(res.drop(columns="protocol").round(3).to_string(index=False))

    # Our out-of-sample numbers must equal what the main pipeline already reported.
    ours = res[res["model"].str.startswith("Ours")].set_index("scored_on")
    out = ours[ours["sample"] == "out-of-sample"]["rmse_all"]
    assert np.isclose(out[2016], pd.read_csv(REPORTS / "final_results.csv")["two_stage"][0])
    assert np.isclose(out[2020], pd.read_csv(REPORTS / "final_results_2020.csv")["two_stage"][0])

    # ---- the paper's Table 4 countries, in 2012 ----
    actual = games[2012].set_index("noc")["medals"]
    t4 = pd.DataFrame({"noc": list(PAPER_TABLE4)})
    t4["actual_2012"] = t4["noc"].map(actual)
    t4["paper_prediction"] = t4["noc"].map(PAPER_TABLE4)
    t4["ours_in_sample"] = t4["noc"].map(preds_2012[(list(models)[0], True)])
    t4["paper_pipeline_in_sample"] = t4["noc"].map(preds_2012[(list(models)[1], True)])
    t4["ours_out_of_sample"] = t4["noc"].map(preds_2012[(list(models)[0], False)])
    cols = ["paper_prediction", "ours_in_sample", "paper_pipeline_in_sample", "ours_out_of_sample"]
    mae = {c: (t4[c] - t4["actual_2012"]).abs().mean() for c in cols}
    t4 = pd.concat([t4, pd.DataFrame([{"noc": "MAE (5 countries)", "actual_2012": np.nan, **mae}])],
                   ignore_index=True)
    t4.to_csv(REPORTS / "paper_protocol_table4.csv", index=False)
    print("\nPaper Table 4 countries, 2012 Games:")
    print(t4.round(2).to_string(index=False))
    print(f"(MAE of the paper's predictions: {PAPER_TABLE4_MAE} against the actuals in its own "
          f"Table 4, {mae['paper_prediction']:.1f} against our 2012 actuals)")


if __name__ == "__main__":
    main()
