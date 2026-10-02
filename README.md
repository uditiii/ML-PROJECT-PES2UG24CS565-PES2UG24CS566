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
  figures/                   EDA, classifier, tuning and final-test plots
src/
  data_prep.py               raw data -> olympics_features.csv
  features.py                FEATURE_COLS, year splits, CSV loader (shared)
  eda.py                     exploratory figures
  train_classifier.py        Stage 1: train + pick the medal / no-medal classifier
  metrics.py                 rmse_all, rmse_top10, selection_score
  pipeline.py                TwoStageModel (Stage 1 + Stage 2)
  train_regressor.py         Stage 2: baseline, tuning, model choice on 2012
  evaluate.py                final 2016 test, error breakdown, saves the final model
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

## Run order

```bash
python src/data_prep.py          # -> data/processed/olympics_features.csv
python src/train_classifier.py   # -> models/classifier.joblib
python src/train_regressor.py    # -> models/regressor_template.joblib (about 1 min)
python src/evaluate.py           # -> 2016 results, models/final_two_stage.joblib
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
  Rankings on 2012 and 2016 differ a lot.
- **No host-country feature:** host nations' inflated team sizes mislead the model (Brazil 2016).
- **Shuffled tuning folds:** tuning uses shuffled 5-fold cross-validation, so tuning folds mix
  years. Reported numbers come only from the year-based 2012 and 2016 splits.
- **Gap to the paper:** we did not reproduce the paper's 2016 result. The data sources and
  cleaning differ, and we haven't determined which difference matters.

## Reference

Dobkowski, "2020 Summer Olympics Predictions Using Machine Learning", Stanford University
course project.
