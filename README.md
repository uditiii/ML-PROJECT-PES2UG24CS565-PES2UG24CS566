# Olympic Medal Prediction with a Two-Stage Model

Predicts how many Summer Olympics medals each country wins from economic and demographic
indicators (GDP, population, team size, past medals). It reproduces the two-stage approach of
Dobkowski's "2020 Summer Olympics Predictions Using Machine Learning" (Stanford): a classifier
decides *any medal?*, then a regressor predicts *how many*.

Course: UE24CS352A.

## Team

- **Person A:** data pipeline (`data_prep.py`, `features.py`) and Stage 1 classifier
  (`train_classifier.py`)
- **Person B:** Stage 2 regressor, two-stage pipeline, evaluation, demo app

## Repo structure

```
data/
  raw/                       Kaggle + World Bank downloads (gitignored, see "Data")
  processed/olympics_features.csv   one row per country per Games, 1988-2016
  processed/tokyo2020_features.csv  the same columns for Tokyo 2020 (second test year)
models/
  classifier.joblib          Stage 1: scaled LogisticRegression, trained 1988-2008
  regressor_template.joblib  Stage 2: chosen tuned regressor (unfitted settings)
  final_two_stage.joblib     both stages trained on 1988-2012; used by the app
reports/
  dropped_rows.csv           rows removed in data prep, with the reason
  regressor_results.csv      Stage 2 models on 2012, before/after tuning
  chosen_model.json          chosen regressor + hyperparameters
  predictions_2012.csv       validation predictions, every country
  predictions_2016.csv       final-test predictions (two-stage and baseline)
  final_results.csv          2016 RMSE: baseline vs two-stage
  top5_2016.csv              top 5 countries, actual vs predicted
  dropped_rows_2020.csv      Tokyo 2020 countries removed in prep, with the reason
  predictions_2020.csv       Tokyo 2020 predictions (two-stage and baseline)
  final_results_2020.csv     2020 RMSE: baseline vs two-stage
  top10_2020.csv             top 10 countries in 2020, actual vs predicted
  paper_*.csv                supplementary paper experiments (see that section)
  figures/                   EDA, classifier, tuning, final-test and paper-experiment plots
src/
  data_prep.py               raw data -> olympics_features.csv
  features.py                FEATURE_COLS, year splits, CSV loader (shared)
  eda.py                     exploratory figures
  train_classifier.py        Stage 1: train + pick the medal / no-medal classifier
  metrics.py                 rmse_all, rmse_top10, selection_score
  pipeline.py                TwoStageModel (Stage 1 + Stage 2)
  train_regressor.py         Stage 2: baseline, tuning, model choice on 2012
  evaluate.py                final 2016 test, error breakdown, saves the final model
  prep_2020.py               Tokyo 2020 raw data -> tokyo2020_features.csv
  predict_2020.py            second test: same model retrained on 1988-2016, predicts 2020
  paper_experiments.py       supplementary: the paper's other experiments
  paper_protocol.py          supplementary: in-sample comparison under the paper's protocol
app/app.py                   Streamlit demo
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Quick start (no downloads needed)

The processed data, trained models and all results are committed, so after setup you can
run the demo straight away:

```bash
streamlit run app/app.py
```

The modelling scripts also run without any raw data: `train_regressor.py`, `evaluate.py`,
`predict_2020.py` and `paper_experiments.py` read only `data/processed/` and `models/`.
Re-running them reproduces the committed results exactly. Only `data_prep.py`, `eda.py` and
`prep_2020.py` need the raw downloads below.

`scikit-learn` is pinned to 1.6.1 because the saved models were pickled with it. You may see
`OptimizeWarning: Unknown solver options: iprint`. That's a harmless mismatch between
scikit-learn 1.6.1 and newer SciPy, and the scripts hide it.

## Data

Put the raw files in `data/raw/`, which is gitignored:

- Kaggle, "120 years of Olympic history: athletes and results":
  https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
  (`athlete_events.csv`, `noc_regions.csv`)
- World Bank World Development Indicators:
  https://databank.worldbank.org/source/world-development-indicators.
  `data_prep.py` downloads these automatically with the `wbgapi` package and caches them as
  `data/raw/wdi.csv`.
- Kaggle, "2021 Olympics in Tokyo", for the 2020 test year:
  https://www.kaggle.com/datasets/arjunprasadsarkhel/2021-olympics-in-tokyo.
  Unzip it into `data/raw/tokyo2020/`. Only `Medals.xlsx` and `Athletes.xlsx` are used.
  `prep_2020.py` pulls the 2017–2020 World Bank values itself, into
  `data/raw/wdi_2017_2020.csv`.

## Run order

```bash
python src/data_prep.py          # -> data/processed/olympics_features.csv (needs raw data)
python src/eda.py                # -> EDA figures (needs data_prep.py to have run first)
python src/train_classifier.py   # -> models/classifier.joblib
python src/train_regressor.py    # -> models/regressor_template.joblib (about 1 min)
python src/evaluate.py           # -> 2016 results, models/final_two_stage.joblib
python src/prep_2020.py          # -> data/processed/tokyo2020_features.csv (needs raw data)
python src/predict_2020.py       # -> Tokyo 2020 results
python src/paper_experiments.py  # optional, supplementary (about 2 min)
python src/paper_protocol.py     # optional, comparison under the paper's protocol
streamlit run app/app.py         # demo
```

**Data split (by Games year, never random):**
- Train and tune on 1988–2008.
- Choose models on 2012.
- Retrain on 1988–2012 and predict 2016 once, as the final test.

## Results

**Final test, 2016** (from `reports/final_results.csv`). RMSE is in medals; lower is better.

| | Baseline (one LinearRegression) | Two-stage (LogReg + RandomForest) | Change |
|---|---|---|---|
| RMSE, all countries | 3.42 | 3.63 | 6.0% worse |
| RMSE, top 10 countries | 9.88 | 10.53 | 6.5% worse |

**Validation, 2012** (used to choose the model):

| | Baseline | Two-stage |
|---|---|---|
| RMSE, all countries | 3.20 | 2.51 |
| RMSE, top 10 countries | 10.32 | 8.12 |

The chosen Stage 2 regressor is a RandomForest (500 trees, `max_depth=20`,
`max_features="sqrt"`, `min_samples_leaf=1`). It had the lowest `MSE_all + 0.25 × MSE_top10`
on 2012.

The two-stage model clearly beat the baseline on 2012 but not on 2016. For comparison, the
paper reports 3.25 (baseline) and 2.26 (final) on 2016. We did not reproduce that gap.

**Why it's worse on 2016:**
- **Stage 1 is not the cause.** The classifier reaches 92% accuracy on 2016, and 98% of the
  squared error comes from Stage 2's medal counts.
- **Brazil, the 2016 host, is the largest miss** (predicted 42, actual 19). Hosts get places in
  nearly every event, so Brazil's team almost doubled (248 → 462 athletes), but its medals
  didn't.
- **Random forests can't predict beyond the range they saw.** The forest tops out near 106,
  so the USA (121) is under-predicted.

## Tokyo 2020 (second test year)

After the model was chosen (on 2012) and tested (on 2016), we added Tokyo 2020 as a second
test. The **same** two-stage model and baseline were retrained on 1988–2016 and predicted 2020
once. Nothing about the model, features or hyperparameters changed after seeing 2020. Results
are from `reports/final_results_2020.csv`.

| | Baseline (one LinearRegression) | Two-stage (LogReg + RandomForest) | Change |
|---|---|---|---|
| RMSE, all countries | 3.43 | 3.01 | 12.3% better |
| RMSE, top 10 countries | 11.67 | 10.16 | 12.9% better |

- **Data:** 197 of 206 Tokyo countries are kept, holding 98.7% of the 1,080 medals. Dropped
  rows are in `reports/dropped_rows_2020.csv`; Chinese Taipei (12 medals, no World Bank data)
  is the largest loss.
- **Stage 1:** accuracy 0.873, F1 0.854. 93% of the squared error is Stage 2's counts.
- **Biggest misses:**
  - ROC/Russia, predicted 54 vs 71 actual. Its 2016 count was cut by the doping ban.
  - Netherlands, predicted 20 vs 36. It nearly doubled its 2016 medals.
  - China, predicted 77 vs 88.
  - Host Japan was close (53 vs 58): its team grew from 335 to 585 athletes *and* its medals
    rose, so the team-size signal that misled us on Brazil 2016 worked here.
- **Context:** Tokyo 2020 was postponed to 2021 and held under COVID restrictions with no
  spectators. Its GDP features are 2020 World Bank values, which reflect the pandemic dip.

So the two-stage model was better on 2012, slightly worse on 2016 and clearly better on 2020.
With one result per Games, the sensible conclusion is "usually better, not always".

## Paper experiments (supplementary)

`python src/paper_experiments.py` re-runs the paper's other ideas with our data, split and
metrics. It writes only new files: `reports/paper_experiments.csv`,
`reports/paper_classifiers.csv` (paper Table 3), `reports/paper_top5_2016.csv` (paper
Table 4), `reports/paper_lwlr_grid.csv` and `reports/figures/paper_experiments.png`.

RMSE is in medals, for all countries. The selection score is the one we used to pick our model.

| Variant | 2012 RMSE | 2012 selection | 2016 RMSE | 2020 RMSE |
|---|---|---|---|---|
| Baseline: one LinearRegression | 3.20 | 36.9 | 3.42 | 3.43 |
| **Chosen: LogisticRegression + RandomForest** | **2.51** | **22.8** | **3.63** | **3.01** |
| Paper's pipeline: GaussianNB + Ridge | 2.98 | 28.9 | 3.51 | 3.49 |
| Locally weighted LinearRegression, paper setting (y0=50, τ=1) | 13.04 | 600.2 | 34.33 | 41.34 |
| Two-stage LinearRegression + log(athletes) | 3.67 | 53.5 | 3.53 | 3.95 |
| SVR, sigmoid kernel (tuned) | 9.53 | 518.1 | 9.65 | 11.22 |
| RandomForest predicting *share* of medals | 2.33 | 16.7 | 3.70 | 2.90 |

GaussianNB is given standardised inputs in these scripts. sklearn adds a smoothing term equal
to `var_smoothing` × the *largest* feature variance (GDP's, about 8×10²³) to every feature's
variance, which drowns small-scale features like `pct_athletes` unless the inputs are scaled
first (2012 recall 0.57 unscaled vs 0.78 scaled).

**What matched the paper:**
- **Log feature maps** didn't help: every log variant has a worse 2012 selection score than
  raw features.

**Classifier choice:** once its inputs are scaled, GaussianNB is competitive. On 2012 it scores
F1 0.83, against 0.83–0.86 for the other four. Swapping our Stage 1 for any of them changes the
medal error very little.

**What didn't match:**
- **The paper's final pipeline** (GaussianNB + Ridge) scores 2.98 / 3.51 / 3.49 on
  2012 / 2016 / 2020, far from the paper's reported 2.26 on 2016.
  - It is worse than our model on 2012 and 2020, but better on 2016 (3.51 vs 3.63).
  - It beats the baseline on 2012 but not on 2016 or 2020.
- **The paper's locally weighted setting** (y0=50, τ=1) leaves an effective 3.2 of 409
  training rows and fails. The best weighting on our grid is the widest (τ=25), and even that
  is worse than no weighting at all.
- **Medal share vs count:** the paper preferred predicting counts, but a RandomForest
  predicting each country's *share* of medals beat our chosen model on 2012 (selection 16.7 vs
  22.8) and on 2020.
  - We still don't switch. Our model was chosen on 2012 before this candidate existed, and
    we'd now be choosing with knowledge of 2016 and 2020, i.e. on the test set.
  - It is also worse on 2016 (3.70 vs 3.63).

**Every variant that beats our chosen model somewhere** (from `paper_experiments.csv`):
- **2012 selection score:** the share-target RandomForest (16.7 vs 22.8) and RandomForest +
  RandomForest (22.75 vs 22.79, effectively a tie).
- **2016 RMSE:** nine variants. They include our tuned linear SVR (2.72), two-stage Ridge
  (3.50), the paper's GaussianNB + Ridge (3.51), two-stage LinearRegression (3.44), two log
  variants, and GaussianNB + RandomForest and MLP + RandomForest (both 3.62, ties).
- **2020 RMSE:** the share-target RandomForest (2.90) and SVC + RandomForest (3.00, a tie).
- **All nine 2016 winners do worse than our model on 2020** (3.05–3.95 vs 3.01). Rankings on a
  single Games don't carry over to the next one, which is exactly why we choose on validation
  and don't switch after seeing a test year.

## Comparison under the paper's protocol

**Why the paper's protocol looks in-sample:**
- The paper's Table 4 "2016 actual" medals are the **2012** results: USA 103, UK 65, GER 44,
  JPN 38. The real 2016 counts are 121, 67, 42 and 41.
- The paper also says its final model was trained on 1988–2012.
- So its reported numbers look like 2012 predictions from a model that had already seen 2012
  in training. That is **in-sample**: the model is scored on its own training data.

`python src/paper_protocol.py` reproduces that protocol *for comparison only*. It writes
`reports/paper_protocol.csv` and `reports/paper_protocol_table4.csv`.

RMSE, all countries, in medals:

| | **In-sample** (paper protocol: train 1988–2012, score 2012) | Out-of-sample: 2012 (train 1988–2008) | Out-of-sample: 2016 | Out-of-sample: 2020 |
|---|---|---|---|---|
| Ours: LogisticRegression + RandomForest | *1.21* | **2.51** | **3.63** | **3.01** |
| Paper's pipeline: GaussianNB + Ridge | *2.75* | 2.98 | 3.51 | 3.49 |
| Baseline: one LinearRegression | *2.79* | 3.20 | 3.42 | 3.43 |

The paper's Table 4 countries in 2012. In-sample = trained on 1988–2012, including 2012.

| | Actual | Paper's prediction | Ours, *in-sample* | Paper pipeline, *in-sample* | Ours, out-of-sample |
|---|---|---|---|---|---|
| USA | 103 | 103 | 102 | 110 | 102 |
| China | 88 | 85 | 83 | 90 | 73 |
| UK | 65 | 52 | 59 | 51 | 52 |
| Germany | 44 | 44 | 42 | 42 | 40 |
| Japan | 38 | 45 | 37 | 36 | 35 |
| **MAE** | | 4.6 (paper reports 4.8) | *3.0* | *5.4* | 7.2 |

The paper's predictions give MAE 4.6 against our 2012 counts, versus the 4.8 it reports. A
one-medal difference in one country's count would explain the gap.

**Why our out-of-sample numbers are the real test:**
- An in-sample score measures how well a model *remembers* data it was trained on, not how
  well it predicts a Games it hasn't seen.
- **The flexible model benefits most.** Scoring 2012 in-sample instead of out-of-sample lowers
  every model's error, but not equally:
  - our random forest: 2.51 → 1.21 (52% lower), because a deep forest can partly memorise
    its training rows;
  - the baseline: 3.20 → 2.79 (13% lower);
  - the simpler GaussianNB + Ridge: 2.98 → 2.75 (8% lower).
- Under the paper's protocol, the paper's pipeline scores 2.75 on our data, closer to its
  reported 2.26 but still not matching it.
- In-sample comparisons therefore favour exactly the models most likely to overfit.
- Under the paper's own protocol, our model's in-sample error (1.21) is far below the paper's
  reported 2.26, but that's not a fair win either.
- The fair comparison is out-of-sample, every Games scored by a model that never saw it:
  2.51 (2012), 3.63 (2016) and 3.01 (2020).

## Known limitations

- **Successor states:** ex-Soviet and ex-Yugoslav successor states (other than Russia, Serbia,
  Germany and the Czech Republic, which are mapped) have `medals_last_games = 0` after the
  split. Kaggle doesn't record which republic Unified Team athletes came from.
- **Missing countries:** countries without World Bank data (e.g. North Korea, Kosovo, South
  Sudan) are excluded. See `reports/dropped_rows.csv`.
- **Early Games are incomplete:** 1988 and 1992 keep only about 55% / 83% of their medals,
  because the USSR, both Germanys and the Unified Team have no World Bank data.
- **1988's `medals_last_games`** comes from the boycotted 1984 Games.
- **One validation year:** models are chosen on 2012 alone, which is a noisy signal.
  Rankings differ a lot between 2012, 2016 and 2020; the 2020 result reverses 2016's.
- **No host-country feature:** a host's inflated team size can mislead the model. Brazil 2016
  got a bigger team but few extra medals (a big miss); Japan 2020 got both (a close prediction).
- **Shuffled tuning folds:** tuning uses shuffled 5-fold cross-validation, so tuning folds mix
  years. Reported numbers come only from the year-based 2012 and 2016 splits.
- **Gap to the paper:** we did not reproduce the paper's 2016 result. The data sources and
  cleaning differ, and we haven't determined which difference matters.

## Reference

Dobkowski, "2020 Summer Olympics Predictions Using Machine Learning", Stanford University
course project.
