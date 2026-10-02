"""EDA figures -> reports/figures/.  Run after data_prep.py:  python src/eda.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features import FEATURE_COLS, load_data, ROOT

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    # 1. null heatmap BEFORE dropping rows
    raw = pd.read_csv(ROOT / "data" / "interim" / "merged_with_nulls.csv")
    plt.figure(figsize=(11, 6))
    sns.heatmap(raw.isna(), cbar=False, yticklabels=False)
    plt.title("Missing values before dropping rows (light = null)")
    plt.tight_layout(); plt.savefig(FIG / "null_heatmap.png", dpi=150); plt.close()

    df = load_data()

    # 2. each feature vs medals
    cols, n = FEATURE_COLS, 4
    nrows = (len(cols) + n - 1) // n
    fig, axes = plt.subplots(nrows, n, figsize=(16, 3.5 * nrows))
    for ax, c in zip(axes.ravel(), cols):
        ax.scatter(df[c], df["medals"], s=8, alpha=0.4)
        ax.set_xlabel(c); ax.set_ylabel("medals")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "features_vs_medals.png", dpi=150); plt.close()

    # 3. class balance
    counts = df["won_medal"].value_counts().sort_index()
    plt.figure(figsize=(4.5, 4))
    ax = sns.barplot(x=["No medal (0)", "Medal (1)"], y=counts.values)
    ax.bar_label(ax.containers[0])
    plt.title("Class balance: won_medal")
    plt.tight_layout(); plt.savefig(FIG / "class_balance.png", dpi=150); plt.close()
    print("Saved figures to", FIG)


if __name__ == "__main__":
    main()
