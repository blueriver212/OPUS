"""
Classify commercial satellites from the UCS database into OPUS species (S, Su, Sns)
and compare populations pre-2016 versus the full historical record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import re


CATEGORY_ORDER: Final[list[str]] = ["S", "Su", "Sns"]
LEO_CODE: Final[str] = "LEO"
SPACE_X_KEYWORD: Final[str] = "SPACEX"
COMMERCIAL_KEYWORD: Final[str] = "Commercial"
CONSTELLATION_KEYWORDS: Final[list[str]] = [
    "IRIDIUM",
    "GLOBALSTAR",
    "DOVE",
    "CORVUS",
    "CAPELLA",
    "BRO",
    "BLACKSKY",
    "ASTROCAST",
    "AEROCUBE",
    "ECHOSTAR",
    "ORBCOMM",
]
CONSTELLATION_REGEX: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(keyword) for keyword in CONSTELLATION_KEYWORDS),
    flags=re.IGNORECASE,
)
PERIGEE_COL: Final[str] = "Perigee (km)"
APOGEE_COL: Final[str] = "Apogee (km)"
ECCENTRICITY_COL: Final[str] = "Eccentricity"
MEAN_ALTITUDE_COL: Final[str] = "mean_altitude_km"
EARTH_RADIUS_KM: Final[float] = 6378.136
PRE_2016_CUTOFF: Final[pd.Timestamp] = pd.Timestamp("2017-01-01")


def _compute_mean_altitude(
    perigee: float | None, apogee: float | None, eccentricity: float | None
) -> float | None:
    """Estimate a representative altitude (km) using available orbital parameters."""
    perigee_val = perigee if pd.notna(perigee) else None
    apogee_val = apogee if pd.notna(apogee) else None
    ecc = float(eccentricity) if pd.notna(eccentricity) else None

    if perigee_val is None and apogee_val is None:
        return None

    if ecc is not None:
        if perigee_val is None and apogee_val is not None and ecc < 1:
            ra = apogee_val + EARTH_RADIUS_KM
            semi_major = ra / (1 + ecc)
            perigee_val = semi_major * (1 - ecc) - EARTH_RADIUS_KM
        elif apogee_val is None and perigee_val is not None and ecc < 1:
            rp = perigee_val + EARTH_RADIUS_KM
            semi_major = rp / (1 - ecc)
            apogee_val = semi_major * (1 + ecc) - EARTH_RADIUS_KM

    values = [val for val in (perigee_val, apogee_val) if val is not None]
    if not values:
        return None
    if len(values) == 2:
        return float(sum(values) / 2.0)
    return float(values[0])


def load_satellite_catalog(csv_path: Path) -> pd.DataFrame:
    """Read the UCS catalogue and standardise the fields needed for classification."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", engine="python")
    df["Launch Mass (kg.)"] = pd.to_numeric(df["Launch Mass (kg.)"], errors="coerce")
    df["Class of Orbit"] = (
        df["Class of Orbit"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"LEO ": LEO_CODE})
    )
    df["Date of Launch"] = pd.to_datetime(df["Date of Launch"], errors="coerce")
    df["Users"] = df["Users"].astype(str)
    df[PERIGEE_COL] = pd.to_numeric(df.get(PERIGEE_COL), errors="coerce")
    df[APOGEE_COL] = pd.to_numeric(df.get(APOGEE_COL), errors="coerce")
    df[ECCENTRICITY_COL] = pd.to_numeric(df.get(ECCENTRICITY_COL), errors="coerce")
    df[MEAN_ALTITUDE_COL] = [
        _compute_mean_altitude(perigee, apogee, ecc)
        for perigee, apogee, ecc in zip(df[PERIGEE_COL], df[APOGEE_COL], df[ECCENTRICITY_COL])
    ]
    df[MEAN_ALTITUDE_COL] = pd.to_numeric(df[MEAN_ALTITUDE_COL], errors="coerce")
    return df


def filter_to_commercial(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows tagged as commercial operators/users."""
    mask = df["Users"].str.contains(COMMERCIAL_KEYWORD, case=False, na=False)
    return df.loc[mask].copy()


def filter_to_leo(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only satellites operating in LEO."""
    return df.loc[df["Class of Orbit"] == LEO_CODE].copy()


def classify_species(df: pd.DataFrame) -> pd.DataFrame:
    """Classify satellites into OPUS species using configured keyword rules."""
    df = df.copy()

    def build_combined_name(row: pd.Series) -> str:
        parts = [
            row.get("Current Official Name of Satellite"),
            row.get("Name of Satellite, Alternate Names"),
        ]
        return " ".join(str(part) for part in parts if isinstance(part, str))

    def classify_row(row: pd.Series) -> str:
        mass = row["Launch Mass (kg.)"]
        name = row["combined_name"]

        if CONSTELLATION_REGEX.search(name):
            return "S"
        if pd.notna(mass) and mass <= 20:
            return "Sns"
        return "Su"

    df["combined_name"] = df.apply(build_combined_name, axis=1)
    df["matches_constellation_keyword"] = df["combined_name"].apply(
        lambda name: bool(CONSTELLATION_REGEX.search(name)) if isinstance(name, str) else False
    )
    df["species_class"] = df.apply(classify_row, axis=1)
    df["is_constellation"] = df["Operator/Owner"].str.contains(
        SPACE_X_KEYWORD, case=False, na=False
    )
    return df


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of species counts before 2016 and across all years."""
    pre_mask = df["Date of Launch"].notna() & (df["Date of Launch"] < PRE_2016_CUTOFF)
    counts_all = df["species_class"].value_counts()
    counts_pre = df.loc[pre_mask, "species_class"].value_counts()

    counts = (
        pd.DataFrame({"Pre-2016": counts_pre, "All Data": counts_all})
        .reindex(CATEGORY_ORDER)
        .fillna(0)
        .astype(int)
    )
    counts["Post-2015"] = counts["All Data"] - counts["Pre-2016"]
    return counts


def prepare_yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate yearly launch counts per species."""
    df = df[df["Date of Launch"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=CATEGORY_ORDER, dtype=int)
    df["LaunchYear"] = df["Date of Launch"].dt.year
    yearly = (
        df.groupby("LaunchYear")["species_class"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
        .sort_index()
    )
    return yearly.astype(int)


def plot_counts(counts: pd.DataFrame, output_path: Path) -> None:
    """Render and persist the grouped bar chart highlighting count differences."""
    ax = counts[["Pre-2016", "All Data"]].plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Number of satellites")
    ax.set_xlabel("Species class")
    ax.set_title("Commercial satellite counts by OPUS species")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved bar chart to {output_path}")
    try:
        plt.show()
    except Exception as exc:  # pragma: no cover - headless environments
        print(f"Plot display skipped: {exc}")


def plot_time_series(
    yearly_all: pd.DataFrame, yearly_commercial: pd.DataFrame, output_path: Path
) -> None:
    """Plot side-by-side time series comparing all satellites vs commercial only."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for col in CATEGORY_ORDER:
        axes[0].plot(
            yearly_all.index,
            yearly_all[col],
            label=col,
            linewidth=2,
        )
    axes[0].set_title("All satellites")
    axes[0].set_xlabel("Launch year")
    axes[0].set_ylabel("Number of satellites")
    axes[0].grid(axis="both", linestyle="--", alpha=0.4)

    for col in CATEGORY_ORDER:
        axes[1].plot(
            yearly_commercial.index,
            yearly_commercial[col],
            label=col,
            linewidth=2,
        )
    axes[1].set_title("Commercial satellites")
    axes[1].set_xlabel("Launch year")
    axes[1].grid(axis="both", linestyle="--", alpha=0.4)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(CATEGORY_ORDER))
    fig.suptitle("Satellite launches by OPUS species over time")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=300)
    print(f"Saved time series plot to {output_path}")
    try:
        plt.show()
    except Exception as exc:  # pragma: no cover - headless environments
        print(f"Plot display skipped: {exc}")


def write_outputs(
    leo_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    counts: pd.DataFrame,
    yearly_all: pd.DataFrame,
    yearly_commercial: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist the enriched catalogue and the aggregated counts."""
    leo_classified_path = output_dir / "ucs_leo_classified.csv"
    classified_path = output_dir / "ucs_commercial_classified.csv"
    shorthand_classified_path = output_dir / "classified.csv"
    counts_path = output_dir / "commercial_counts_summary.csv"
    yearly_all_path = output_dir / "yearly_counts_all.csv"
    yearly_commercial_path = output_dir / "yearly_counts_commercial.csv"

    leo_df.to_csv(leo_classified_path, index=False)
    commercial_df.to_csv(classified_path, index=False)
    commercial_df.to_csv(shorthand_classified_path, index=False)
    counts.to_csv(counts_path)
    yearly_all.to_csv(yearly_all_path)
    yearly_commercial.to_csv(yearly_commercial_path)
    print(f"Saved LEO classified catalogue to {leo_classified_path}")
    print(f"Saved classified catalogue to {classified_path}")
    print(f"Saved classified catalogue copy to {shorthand_classified_path}")
    print(f"Saved summary table to {counts_path}")
    print(f"Saved yearly counts (all) to {yearly_all_path}")
    print(f"Saved yearly counts (commercial) to {yearly_commercial_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "UCS_Satellite_db.csv"
    output_dir = Path(__file__).resolve().parent
    plot_path = output_dir / "commercial_satellite_counts.png"
    time_series_plot_path = output_dir / "satellite_counts_time_series.png"

    df = load_satellite_catalog(dataset_path)
    all_classified = classify_species(df)
    leo_classified = filter_to_leo(all_classified)
    commercial_df = filter_to_commercial(leo_classified)
    counts = summarize_counts(commercial_df)
    print("\nCommercial satellite counts (Pre-2016 vs All Data):\n")
    print(counts)

    yearly_all = prepare_yearly_counts(leo_classified)
    yearly_commercial = prepare_yearly_counts(commercial_df)

    write_outputs(leo_classified, commercial_df, counts, yearly_all, yearly_commercial, output_dir)
    plot_counts(counts, plot_path)
    plot_time_series(yearly_all, yearly_commercial, time_series_plot_path)


if __name__ == "__main__":
    main()
