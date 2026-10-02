"""Raw Kaggle + World Bank data -> data/processed/olympics_features.csv

Run:  python src/data_prep.py
Needs data/raw/athlete_events.csv and data/raw/noc_regions.csv (Kaggle).
World Bank data is pulled with wbgapi and cached in data/raw/wdi.csv.
"""
from pathlib import Path
import pandas as pd

from features import CSV_COLS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FIRST_YEAR, LAST_YEAR = 1988, 2016

WB_CODES = {
    "NY.GDP.MKTP.CD": "gdp",
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "SP.POP.TOTL": "population",
    "SP.POP.GROW": "pop_growth",
    "AG.LND.TOTL.K2": "land_area",
}

# IOC code -> ISO3 where they differ. Anything not listed is assumed identical.
NOC_TO_ISO3 = {
    "ALG": "DZA", "ANG": "AGO", "ANT": "ATG", "ARU": "ABW", "ASA": "ASM",
    "BAH": "BHS", "BAN": "BGD", "BAR": "BRB", "BER": "BMU", "BHU": "BTN",
    "BIZ": "BLZ", "BOT": "BWA", "BRN": "BHR", "BRU": "BRN", "BUL": "BGR",
    "BUR": "BFA", "CAM": "KHM", "CAY": "CYM", "CGO": "COG", "CHA": "TCD",
    "CHI": "CHL", "CRC": "CRI", "CRO": "HRV", "DEN": "DNK", "ESA": "SLV",
    "FIJ": "FJI", "GAM": "GMB", "GBS": "GNB", "GEQ": "GNQ", "GER": "DEU",
    "GRE": "GRC", "GRN": "GRD", "GUA": "GTM", "GUI": "GIN", "HAI": "HTI",
    "HON": "HND", "INA": "IDN", "IRI": "IRN", "ISV": "VIR", "IVB": "VGB",
    "KOS": "XKX", "KSA": "SAU", "KUW": "KWT", "LAT": "LVA", "LBA": "LBY",
    "LES": "LSO", "LIB": "LBN", "MAD": "MDG", "MAS": "MYS", "MAW": "MWI",
    "MGL": "MNG", "MON": "MCO", "MRI": "MUS", "MTN": "MRT", "MYA": "MMR",
    "NCA": "NIC", "NED": "NLD", "NEP": "NPL", "NGR": "NGA", "NIG": "NER",
    "OMA": "OMN", "PAR": "PRY", "PHI": "PHL", "PLE": "PSE", "POR": "PRT",
    "PUR": "PRI", "RSA": "ZAF", "SAM": "WSM", "SEY": "SYC", "SIN": "SGP",
    "SKN": "KNA", "SLO": "SVN", "SOL": "SLB", "SRI": "LKA", "SUD": "SDN",
    "SUI": "CHE", "TAN": "TZA", "TGA": "TON", "TOG": "TGO", "UAE": "ARE",
    "URU": "URY", "VAN": "VUT", "VIE": "VNM", "VIN": "VCT", "ZAM": "ZMB",
    "ZIM": "ZWE",
    # Serbia and Montenegro 1996-2004: World Bank has no SCG series, so use SRB.
    # Note SRB figures cover Serbia only (no Montenegro), so they understate a bit.
    "SCG": "SRB",
}
# Deliberately dropped (no World Bank series): URS, EUN, GDR, FRG, TCH, YUG,
# TPE, AHO, COK, IOA, EOR, ROT, etc. They simply won't match and get reported.

# medals_last_games when the NOC code changed between Games: (successor, year) ->
# predecessor NOCs whose medals at the previous Games are summed. Used ONLY for
# medals_last_games. Known limitation: other ex-Soviet / ex-Yugoslav successors and
# SVK stay at 0, because the Kaggle data doesn't record which republic Unified Team
# (or Czechoslovak / Yugoslav) athletes came from, so their medals can't be split.
PREDECESSORS = {
    ("GER", 1992): ["FRG", "GDR"],
    ("EUN", 1992): ["URS"],
    ("RUS", 1996): ["EUN"],
    ("CZE", 1996): ["TCH"],
    ("SRB", 2008): ["SCG"],
}


def load_olympics() -> pd.DataFrame:
    ev = pd.read_csv(RAW / "athlete_events.csv")
    return ev[(ev["Season"] == "Summer") & (ev["Year"] != 1906)]  # 1906 = unofficial Intercalated Games


def build_olympic_table(ev: pd.DataFrame) -> pd.DataFrame:
    # Base table: every country that sent athletes (so zero-medal countries stay)
    athletes = ev.groupby(["Year", "NOC"])["ID"].nunique().rename("athletes")

    # Team events count once per team: dedupe before counting
    medal_rows = ev[ev["Medal"].notna()].drop_duplicates(["Year", "NOC", "Event", "Medal"])
    medals = medal_rows.groupby(["Year", "NOC"]).size().rename("medals")

    df = athletes.to_frame().join(medals).reset_index()
    df["medals"] = df["medals"].fillna(0).astype(int)

    # Yearly totals over ALL countries (before any World Bank dropping)
    df["total_medals_awarded"] = df.groupby("Year")["medals"].transform("sum")

    # medals_last_games on the FULL history, 0 if the country was absent last Games
    grid = df.pivot(index="NOC", columns="Year", values="medals").fillna(0)
    prev = (grid.shift(1, axis=1).reset_index()
                .melt(id_vars="NOC", var_name="Year", value_name="medals_last_games"))
    df = df.merge(prev, on=["NOC", "Year"], how="left")

    # NOC code changed since last Games: take the predecessors' medals instead of 0
    print("medals_last_games from predecessors:")
    for (noc, year), preds in PREDECESSORS.items():
        prev_year = grid.columns[grid.columns.get_loc(year) - 1]
        prev_medals = grid.loc[preds, prev_year]
        df.loc[(df["NOC"] == noc) & (df["Year"] == year), "medals_last_games"] = prev_medals.sum()
        print(f"  {noc} {year} <- {prev_medals.to_dict()} ({prev_year}) = {prev_medals.sum():.0f}")
    df["medals_last_games"] = df["medals_last_games"].fillna(0).astype(int)

    return df[(df["Year"] >= FIRST_YEAR) & (df["Year"] <= LAST_YEAR)].copy()


def load_wdi() -> pd.DataFrame:
    cache = RAW / "wdi.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import wbgapi as wb
    wdi = wb.data.DataFrame(list(WB_CODES), time=range(1984, 2017),
                            columns="series", numericTimeKeys=True)
    wdi.index.names = ["iso3", "year"]
    wdi = wdi.reset_index()
    wdi["year"] = wdi["year"].astype(str).str.replace("YR", "").astype(int)
    wdi = wdi.rename(columns=WB_CODES)
    wdi.to_csv(cache, index=False)
    return wdi


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = build_olympic_table(load_olympics())
    names = pd.read_csv(RAW / "noc_regions.csv").set_index("NOC")["region"]
    df["country"] = df["NOC"].map(names).fillna(df["NOC"])
    df["iso3"] = df["NOC"].map(lambda n: NOC_TO_ISO3.get(n, n))
    df = df.rename(columns={"Year": "year", "NOC": "noc"})
    print(f"Olympic rows (1988-2016, all countries): {len(df)}")

    # ---- merge with World Bank ----
    wdi = load_wdi()
    known = set(wdi["iso3"])
    unmatched = df[~df["iso3"].isin(known)]
    print("\nUnmatched NOCs (dropped, no World Bank economy):")
    summary = (unmatched.groupby(["noc", "country"])["year"]
               .apply(lambda s: ",".join(map(str, sorted(s)))))
    print(summary.to_string())
    (REPORTS / "dropped_teams.txt").write_text(summary.to_string())

    df = df[df["iso3"].isin(known)].merge(wdi, on=["iso3", "year"], how="left")

    # ---- features ----
    df["won_medal"] = (df["medals"] > 0).astype(int)
    # pct_* columns are added AFTER every drop (below), so leave them out for now
    pct_cols = ["pct_world_gdp", "pct_world_pop", "pct_athletes"]
    df = df[[c for c in CSV_COLS if c not in pct_cols]]

    df.to_csv(INTERIM / "merged_with_nulls.csv", index=False)   # for the null heatmap
    print("\nNulls per column before dropping:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())

    before = len(df)
    df = df.dropna().copy()
    print(f"\nDropped {before - len(df)} rows with nulls ({before} -> {len(df)})")

    # Shares of the countries actually kept, so each year's column sums to 100
    yearly = df.groupby("year")
    df["pct_world_gdp"] = df["gdp"] / yearly["gdp"].transform("sum") * 100
    df["pct_world_pop"] = df["population"] / yearly["population"].transform("sum") * 100
    df["pct_athletes"] = df["athletes"] / yearly["athletes"].transform("sum") * 100
    df = df[CSV_COLS]
    print("\npct column sums per year:\n", df.groupby("year")[pct_cols].sum().round(2).to_string())

    int_cols = ["year", "athletes", "medals_last_games", "total_medals_awarded", "medals", "won_medal"]
    df[int_cols] = df[int_cols].astype(int)
    df.sort_values(["year", "noc"]).to_csv(PROC / "olympics_features.csv", index=False)

    print("\nRows per year:\n", df["year"].value_counts().sort_index().to_string())
    print("\nClass balance:\n", df["won_medal"].value_counts().to_string())
    print(f"\nSaved {PROC / 'olympics_features.csv'}")


if __name__ == "__main__":
    main()
