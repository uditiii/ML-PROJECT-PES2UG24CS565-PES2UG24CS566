"""Streamlit demo.  Run from the repo root:  streamlit run app/app.py

Reads only SAVED results (predictions CSVs, final model). Nothing is retrained
here, so the page loads fast and shows exactly the numbers in reports/.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Paths are built from THIS file's location, not the current folder, so the app
# works on any machine and from any working directory.
ROOT = Path(__file__).resolve().parents[1]
# src/ must be importable: the saved final model is a pipeline.TwoStageModel, and
# unpickling it needs to `import pipeline`. We also reuse metrics and features.
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURE_COLS        # noqa: E402  (import after the path fix)
from metrics import rmse_all, rmse_top10  # noqa: E402

REPORTS = ROOT / "reports"
YEARS = {2012: "2012 (validation)", 2016: "2016 (final test)",
         2020: "Tokyo 2020 (second test, held 2021)"}

st.set_page_config(page_title="Olympic medal predictor", layout="wide")


# ---------- loading (cached: done once, not on every slider move) ----------
@st.cache_data
def load_predictions(year):
    """One table per year with the same columns: actual, predicted, baseline, p_medal."""
    if year == 2012:
        df = pd.read_csv(REPORTS / "predictions_2012.csv")
        # The 2012 baseline wasn't saved per country, only its scores (chosen_model.json).
        df["baseline"] = pd.NA
    else:   # 2016 and 2020 files have the same columns
        df = pd.read_csv(REPORTS / f"predictions_{year}.csv").rename(
            columns={"predicted_two_stage": "predicted", "predicted_baseline": "baseline"})
        df["year"] = year
    df["error"] = df["predicted"] - df["actual"]   # + = over-predicted, - = under
    return df


@st.cache_data
def load_features():
    # 1988-2016 plus the Tokyo 2020 rows (same columns), for the country explorer.
    proc = ROOT / "data" / "processed"
    return pd.concat([pd.read_csv(proc / "olympics_features.csv"),
                      pd.read_csv(proc / "tokyo2020_features.csv")], ignore_index=True)


@st.cache_data
def load_json(name):
    return json.loads((REPORTS / name).read_text())


@st.cache_data
def load_final_results(year):
    name = "final_results.csv" if year == 2016 else f"final_results_{year}.csv"
    return pd.read_csv(REPORTS / name).set_index("metric")


@st.cache_resource   # resource = an object we call methods on (not plain data)
def load_final_model():
    # Trained on 1988-2012 by evaluate.py. Used ONLY for the what-if section.
    return joblib.load(ROOT / "models" / "final_two_stage.joblib")


feats = load_features()
chosen = load_json("chosen_model.json")

# ---------- 1. header ----------
st.title("Predicting Olympic medal counts from economics & demographics")
st.markdown(
    "**Two-stage model:** a classifier first decides *will this country win any medal?*; "
    f"if yes, a regressor ({chosen['model']}, trained only on medal winners) predicts "
    "*how many*. Most countries win 0, so splitting the question stops that pile of "
    "zeros from dragging every prediction down.")

# ---------- 2. year selector + scores + table ----------
year = st.radio("Games", list(YEARS), format_func=YEARS.get, horizontal=True)
pred = load_predictions(year)
scored = pred.rename(columns={"actual": "medals"})   # metrics expect a `medals` column

if year == 2012:
    base_all, base_top = chosen["baseline_2012"]["rmse_all"], chosen["baseline_2012"]["rmse_top10"]
else:
    final_res = load_final_results(year)
    base_all, base_top = final_res.loc["rmse_all", "baseline"], final_res.loc["rmse_top10", "baseline"]
r_all, r_top = rmse_all(scored["medals"], scored["predicted"]), rmse_top10(scored, scored["predicted"])

c1, c2, c3, c4 = st.columns(4)
# delta_color="inverse": a NEGATIVE delta (lower error than baseline) shows green.
c1.metric("RMSE, all countries", f"{r_all:.2f} medals", f"{r_all - base_all:+.2f} vs baseline",
          delta_color="inverse")
c2.metric("RMSE, top 10 countries", f"{r_top:.2f} medals", f"{r_top - base_top:+.2f} vs baseline",
          delta_color="inverse")
c3.metric("Baseline RMSE (one LinearRegression)", f"{base_all:.2f} medals")
c4.metric("Countries", len(pred))
st.caption("Baseline = a single LinearRegression on all countries, no classifier. "
           "Lower RMSE is better. 2012 picked the model; 2016 (trained on 1988-2012) and "
           "Tokyo 2020 (trained on 1988-2016) were each predicted once, with nothing changed "
           "in between.")
if year == 2020:
    st.caption("Tokyo 2020 was held in 2021 under COVID restrictions, without spectators, and "
               "its GDP features are 2020 values that include the pandemic dip.")

show = pred[["country", "noc", "actual", "predicted", "error", "p_medal"]]
if year != 2012:
    show = pred[["country", "noc", "actual", "predicted", "baseline", "error", "p_medal"]]
st.dataframe(
    show.sort_values("actual", ascending=False), hide_index=True, height=320,
    column_config={
        "country": "Country", "noc": "NOC", "actual": "Actual medals",
        "predicted": "Predicted (two-stage)", "baseline": "Predicted (baseline)",
        "error": st.column_config.NumberColumn("Error (pred - actual)", format="%+d"),
        "p_medal": st.column_config.ProgressColumn("Stage 1: P(any medal)", min_value=0,
                                                   max_value=1, format="%.2f"),
    })

# ---------- 3. predicted vs actual scatter ----------
lim = max(pred["actual"].max(), pred["predicted"].max()) * 1.08
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect prediction (y = x)",
                         line=dict(color="#e34948", width=2), hoverinfo="skip"))
fig.add_trace(go.Scatter(
    x=pred["actual"], y=pred["predicted"], mode="markers", name="country",
    marker=dict(size=9, color="#2a78d6", opacity=0.7, line=dict(color="white", width=1)),
    customdata=pred[["country", "error"]],
    hovertemplate="<b>%{customdata[0]}</b><br>actual %{x}, predicted %{y}"
                  "<br>error %{customdata[1]:+d}<extra></extra>"))
fig.update_layout(title=f"Predicted vs actual medals, {YEARS[year]}", height=460,
                  xaxis=dict(title="Actual medals", range=[0, lim]),
                  yaxis=dict(title="Predicted medals (two-stage)", range=[0, lim]),
                  legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=70))
st.plotly_chart(fig)

# ---------- 4. country explorer ----------
st.header("Country explorer")
countries = sorted(pred["country"].unique())
default = countries.index("USA") if "USA" in countries else 0
country = st.selectbox("Country", countries, index=default)
row = pred[pred["country"] == country].iloc[0]
feat_row = feats[(feats["noc"] == row["noc"]) & (feats["year"] == year)]

k1, k2, k3 = st.columns(3)
k1.metric("Stage 1: P(wins any medal)", f"{row['p_medal']:.0%}")
k2.metric("Final prediction", f"{int(row['predicted'])} medals")
k3.metric("Actual", f"{int(row['actual'])} medals")
with st.expander(f"{country}'s features in {year}"):
    st.dataframe(feat_row[FEATURE_COLS].T.rename(columns=lambda _: "value"))

# ---------- 5. what-if sliders (live predictions from the final model) ----------
st.header("What if?")
base_row = feats[(feats["noc"] == row["noc"]) & (feats["year"] == 2016)]
if base_row.empty:
    st.info(f"{country} has no 2016 row in the data, so there's nothing to scale.")
else:
    s1, s2 = st.columns(2)
    gdp_x = s1.slider("GDP multiplier", 0.5, 3.0, 1.0, 0.1)
    ath_x = s2.slider("Athletes multiplier", 0.5, 3.0, 1.0, 0.1)

    model = load_final_model()
    x = base_row.copy()
    # Scale each raw column AND the columns derived from it, by the same factor.
    for col in ["gdp", "gdp_per_capita", "pct_world_gdp"]:
        x[col] = x[col] * gdp_x
    for col in ["athletes", "pct_athletes"]:
        x[col] = x[col] * ath_x
    now = int(model.predict(x)[0])
    start = int(model.predict(base_row)[0])   # same row at 1x, for the delta
    p_now = model.predict_proba_medal(x)[0]

    w1, w2 = st.columns(2)
    w1.metric(f"Predicted 2016 medals for {country}", now, f"{now - start:+d} vs 1x")
    w2.metric("Stage 1: P(wins any medal)", f"{p_now:.0%}")
    st.caption(
        "Approximation: scaling one country's GDP or athletes would also change the world "
        "totals, so pct_world_gdp and pct_athletes are scaled proportionally rather than "
        "recomputed. gdp_per_capita assumes population stays the same. Stage 2 is a random "
        "forest, so predictions move in steps and can't go much above the largest counts "
        "seen in training (about 106 medals).")

# ---------- 6. how it works ----------
with st.expander("How it works"):
    st.markdown(f"""
- **Data:** one row per country per Summer Games, 1988–2016, including countries that won
  0 medals. Features: GDP, GDP per capita and growth, population and growth, land area,
  share of world GDP / population / athletes, team size, medals at the previous Games,
  total medals awarded, and the year.
- **Stage 1 (classifier):** logistic regression (with feature scaling) predicts
  *any medal: yes/no*. Trained on all countries.
- **Stage 2 (regressor):** {chosen['model']} with `{chosen['params']}` predicts the medal
  count. Trained only on countries that won at least one medal.
- **Final prediction:** 0 if Stage 1 says no, otherwise Stage 2's count, rounded and never
  below 0.
- **Split by year, never random:** tune on 1988–2008, choose models on **2012**, then retrain
  on 1988–2012 and predict **2016** once as the final test. **Tokyo 2020** is a second test:
  the same model retrained on 1988–2016, predicted once, nothing changed after seeing it.
- **Model choice:** lowest `MSE_all + 0.25 × MSE_top10` on 2012, so neither the many small
  countries nor the few big ones dominate.
""")
