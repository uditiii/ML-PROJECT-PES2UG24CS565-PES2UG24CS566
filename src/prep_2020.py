"""Tokyo 2020 (held in 2021) -> data/processed/tokyo2020_features.csv
Run:  python src/prep_2020.py

Needs the Kaggle dataset "2021 Olympics in Tokyo" (arjunprasadsarkhel/2021-olympics-in-tokyo)
unzipped in data/raw/tokyo2020/. We use two of its files:
  Medals.xlsx    one row per country: Gold, Silver, Bronze, Total (team medals counted once)
  Athletes.xlsx  one row per athlete per discipline, with the athlete's country
World Bank 2017-2020 values are pulled with wbgapi and cached in data/raw/wdi_2017_2020.csv.

Builds rows with EXACTLY the columns of olympics_features.csv, using the same rules as
data_prep.py (same World Bank indicators, same NOC->ISO3 map, same land_area fill, pct_*
columns computed after all drops). olympics_features.csv itself is not touched.
"""
import warnings

import pandas as pd

from data_prep import NOC_TO_ISO3, PROC, RAW, REPORTS, WB_CODES, load_wdi
from features import CSV_COLS, load_data

# openpyxl complains that the Kaggle workbooks have no default style; harmless.
warnings.filterwarnings("ignore", message="Workbook contains no default style")

YEAR = 2020
TOKYO_RAW = RAW / "tokyo2020"

# The Tokyo files use official country NAMES. Most match a region name (or note) in
# Kaggle's noc_regions.csv, which is how our CSV got its labels. These are the ones that
# don't, mapped by hand to the codes our CSV uses.
NAME_FIXES = {
    "United States of America": "USA", "People's Republic of China": "CHN",
    "Great Britain": "GBR", "ROC": "RUS",   # Russian Olympic Committee = Russia
    "Republic of Korea": "KOR", "Islamic Republic of Iran": "IRI",
    "Chinese Taipei": "TPE", "Hong Kong, China": "HKG", "Bolivia": "BOL",
    "Brunei Darussalam": "BRU", "Congo": "CGO", "Côte d'Ivoire": "CIV",
    "Democratic Republic of Timor-Leste": "TLS", "Eswatini": "SWZ",
    "Federated States of Micronesia": "FSM", "Kuwait": "KUW",
    "Lao People's Democratic Republic": "LAO", "North Macedonia": "MKD",
    "Republic of Moldova": "MDA", "Saint Kitts and Nevis": "SKN",
    "Singapore": "SGP", "St Vincent and the Grenadines": "VIN",
    "Syrian Arab Republic": "SYR", "United Republic of Tanzania": "TAN",
}


def name_to_noc():
    """Country name -> NOC code. Only codes used at the 2016 Games are candidates, so
    "Germany" maps to GER (not the old FRG/GDR) and "Serbia" to SRB (not YUG/SCG)."""
    ev = pd.read_csv(RAW / "athlete_events.csv", usecols=["NOC", "Year", "Season"])
    nocs_2016 = set(ev.loc[(ev["Year"] == 2016) & (ev["Season"] == "Summer"), "NOC"])
    regions = pd.read_csv(RAW / "noc_regions.csv")
    regions = regions[regions["NOC"].isin(nocs_2016)]
    mapping = {}
    for col in ["region", "notes"]:   # notes holds e.g. "Refugee Olympic Team"
        ok = regions[col].notna()
        mapping.update(dict(zip(regions.loc[ok, col], regions.loc[ok, "NOC"])))
    mapping.update(NAME_FIXES)
    return mapping


def load_wdi_2020():
    """World Bank series for 2017-2020, same indicators as data_prep.load_wdi().
    (data_prep's cache, wdi.csv, stops at 2016, so 2020 needs its own pull.)"""
    cache = RAW / "wdi_2017_2020.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import wbgapi as wb
    wdi = wb.data.DataFrame(list(WB_CODES), time=range(2017, YEAR + 1),
                            columns="series", numericTimeKeys=True)
    wdi.index.names = ["iso3", "year"]
    wdi = wdi.reset_index()
    wdi["year"] = wdi["year"].astype(str).str.replace("YR", "").astype(int)
    wdi = wdi.rename(columns=WB_CODES)
    wdi.to_csv(cache, index=False)
    return wdi


def main():
    # ---- Olympic side: medals and athletes per country NAME ----
    medal_table = pd.read_excel(TOKYO_RAW / "Medals.xlsx")
    athletes = pd.read_excel(TOKYO_RAW / "Athletes.xlsx")
    # Athletes.xlsx has one row per athlete PER DISCIPLINE, so count unique names
    # within each country (the "NOC" column there actually holds the country name).
    per_country = athletes.groupby("NOC")["Name"].nunique().rename("athletes").reset_index()
    per_country = per_country.rename(columns={"NOC": "name"})
    medals = medal_table.set_index("Team/NOC")["Total"]
    per_country["medals"] = per_country["name"].map(medals).fillna(0).astype(int)
    assert set(medals.index) <= set(per_country["name"]), "medal country missing from Athletes.xlsx"
    total_medals = int(medal_table["Total"].sum())   # all 2020 medals, before any drop

    # ---- names -> NOC codes ----
    codes = name_to_noc()
    per_country["noc"] = per_country["name"].map(codes)
    unmapped = per_country[per_country["noc"].isna()]
    print("Country names with no NOC code:", unmapped["name"].tolist() or "none")
    dropped = [unmapped.assign(noc=unmapped["name"], year=YEAR, reason="no NOC code for this name")]
    df = per_country[per_country["noc"].notna()].copy()
    assert not df["noc"].duplicated().any(), "two names mapped to the same NOC"

    # ---- columns that come from our existing data ----
    ours = load_data()
    last = ours[ours["year"] == 2016].set_index("noc")
    # medals_last_games = the country's 2016 medals in our CSV. Countries without a 2016
    # row there (e.g. Kuwait competed as Independent Olympic Athletes in 2016) get 0,
    # the same "absent last Games -> 0" rule data_prep.py uses.
    df["medals_last_games"] = df["noc"].map(last["medals"]).fillna(0).astype(int)
    print("No 2016 row in our CSV (medals_last_games = 0):",
          sorted(df.loc[~df["noc"].isin(last.index), "noc"]))
    # Same country label as our CSV where we have one (keeps the app consistent).
    labels = ours.drop_duplicates("noc", keep="last").set_index("noc")["country"]
    df["country"] = df["noc"].map(labels).fillna(df["name"])
    df["year"] = YEAR
    df["total_medals_awarded"] = total_medals

    # ---- World Bank: same matching and land_area rule as data_prep.py ----
    df["iso3"] = df["noc"].map(lambda n: NOC_TO_ISO3.get(n, n))
    # 1984-2016 (data_prep's cache) + 2017-2020, so land_area can be forward-filled
    # from earlier years exactly like data_prep.py does (ffill then bfill per country).
    wdi = pd.concat([load_wdi(), load_wdi_2020()]).sort_values(["iso3", "year"])
    wdi["land_area"] = wdi.groupby("iso3")["land_area"].transform(lambda s: s.ffill().bfill())
    known = set(wdi["iso3"])
    no_wb = df[~df["iso3"].isin(known)]
    dropped.append(no_wb.assign(reason="no World Bank match"))
    df = df[df["iso3"].isin(known)].merge(wdi[wdi["year"] == YEAR], on=["iso3", "year"], how="left")

    # ---- nulls: log, then drop (GDP and population are NOT imputed) ----
    df["won_medal"] = (df["medals"] > 0).astype(int)
    pct_cols = ["pct_world_gdp", "pct_world_pop", "pct_athletes"]
    df = df[[c for c in CSV_COLS if c not in pct_cols]]
    nulls = df.isna()
    has_null = nulls.any(axis=1)
    dropped.append(df.loc[has_null].assign(
        reason=nulls[has_null].apply(lambda r: "missing " + ", ".join(r.index[r]), axis=1)))
    print("\nNulls per column before dropping:")
    print(nulls.sum()[nulls.sum() > 0].to_string())
    df = df[~has_null].copy()

    log = pd.concat(dropped)[["noc", "year", "medals", "reason"]].sort_values("noc")
    log.to_csv(REPORTS / "dropped_rows_2020.csv", index=False)
    print(f"\nDropped {len(log)} countries:")
    print(log.to_string(index=False))

    # ---- pct_* AFTER every drop: shares of the countries kept (as data_prep.py) ----
    df["pct_world_gdp"] = df["gdp"] / df["gdp"].sum() * 100
    df["pct_world_pop"] = df["population"] / df["population"].sum() * 100
    df["pct_athletes"] = df["athletes"] / df["athletes"].sum() * 100
    df = df[CSV_COLS]
    int_cols = ["year", "athletes", "medals_last_games", "total_medals_awarded", "medals", "won_medal"]
    df[int_cols] = df[int_cols].astype(int)
    df.sort_values("noc").to_csv(PROC / "tokyo2020_features.csv", index=False)

    print(f"\nRows kept: {len(df)}  (medal winners: {df['won_medal'].sum()})")
    print(f"Medals kept: {df['medals'].sum()} of {total_medals} "
          f"({df['medals'].sum() / total_medals * 100:.1f}%)")
    print("pct column sums:", df[pct_cols].sum().round(2).to_dict())
    print(f"Saved {PROC / 'tokyo2020_features.csv'}")


if __name__ == "__main__":
    main()
