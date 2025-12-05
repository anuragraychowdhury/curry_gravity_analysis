"""
Analysis: Does playing with Steph Curry improve teammate shooting efficiency?

This script implements a very simple analysis that meets the project requirements:
- Quantitative response variables: three-point percentage, field goal percentage,
  points per game, and assists per game
- Two quantitative predictors: minutes per game (MP), points per game (PTS)
- One categorical predictor: team (plays with Curry vs other teams)
- At least two additional predictors: age (quantitative), position (categorical)

Usage (basic):
    1. Put your stats CSV into the `data/` folder (e.g. nba_per_game_processed.csv).
    2. Make sure `CSV_PATH` and `COLUMN_MAP` below match your file and column names.
    3. Run:
           python analysis/curry_effect.py
"""

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# CONFIGURATION: EDIT THESE TO MATCH YOUR CSV FILES / COLUMN NAMES
# ---------------------------------------------------------------------------

# Path to your stats CSV (put it in the `data/` folder)
# For this project we assume: data/nba_per_game_processed.csv
CSV_PATH = os.path.join("data", "nba_per_game_processed.csv")

# Expected column names in the data
# If your CSV uses different names, change the right-hand side values.
COLUMN_MAP = {
    # Categorical / ID columns
    "team": "Team",               # team name string, e.g. "GSW"
    "position": "Position",       # position string, e.g. "PG", "SG", "PF", etc.
    # Quantitative predictors
    "minutes": "MP",              # minutes per game
    "points": "PTS",              # points per game
    "age": "Age",                 # age in years
    # Performance indicators (responses)
    "three_point_pct": "3P%",     # three-point percentage as decimal or percent
    "fg_pct": "FG%",              # field goal percentage
    "assists": "AST",             # assists per game
    # We can also treat points per game as a response metric as well
    # (in addition to being a predictor).
    # This will be handled carefully so we don't use points as both
    # predictor and response in the same model.
    # You can add more here if they exist in your CSV, e.g.:
    # "efg_pct": "eFG%",
}

# Value that indicates a player is playing with Curry (Golden State Warriors).
# Adjust this if your CSV uses a different label.
CURRY_TEAM_VALUES: List[str] = ["GSW"]


def load_and_prepare_data() -> pd.DataFrame:
    """Load the stats CSV and engineer features for analysis."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV file not found.\n"
            f"Expected: {CSV_PATH}\n\n"
            f"Put your stats CSV into the `data/` folder and update CSV_PATH "
            f"at the top of analysis/curry_effect.py."
        )

    df = pd.read_csv(CSV_PATH)

    # Rename / select relevant columns
    # Required predictors
    try:
        team_col = COLUMN_MAP["team"]
        mp_col = COLUMN_MAP["minutes"]
        pts_col = COLUMN_MAP["points"]
        age_col = COLUMN_MAP["age"]
        pos_col = COLUMN_MAP["position"]
    except KeyError as e:
        raise KeyError(f"Missing column mapping for: {e}. Check COLUMN_MAP.") from e

    # Response columns we will analyze (only keep those that actually exist)
    response_map = {
        "three_point_pct": COLUMN_MAP.get("three_point_pct"),
        "fg_pct": COLUMN_MAP.get("fg_pct"),
        "points": COLUMN_MAP.get("points"),      # PTS per game
        "assists": COLUMN_MAP.get("assists"),    # AST per game
    }
    # Filter out None values
    response_map = {k: v for k, v in response_map.items() if v is not None}

    # Check that required columns exist
    base_cols = [team_col, mp_col, pts_col, age_col, pos_col]
    missing_base = [c for c in base_cols if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Your CSV is missing expected base columns: {missing_base}\n"
            f"Columns found: {sorted(df.columns.tolist())}\n"
            f"Update COLUMN_MAP in analysis/curry_effect.py to match your CSV headers."
        )

    # Check available response columns
    missing_responses = [c for c in response_map.values() if c not in df.columns]
    if len(missing_responses) == len(response_map):
        raise ValueError(
            f"None of the expected response columns were found: {list(response_map.values())}\n"
            f"Columns found: {sorted(df.columns.tolist())}\n"
            f"Update COLUMN_MAP or add appropriate stats columns to your CSV."
        )

    # Keep only the columns we care about
    keep_cols = base_cols + [c for c in response_map.values() if c in df.columns]
    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Your CSVs are missing expected columns: {missing_cols}\n"
            f"Columns found: {sorted(df.columns.tolist())}\n"
            f"Update COLUMN_MAP in analysis/curry_effect.py to match your CSV headers."
        )

    df = df[keep_cols].copy()

    # Standardize names internally
    rename_map = {
        team_col: "team",
        mp_col: "minutes",
        pts_col: "points",
        age_col: "age",
        pos_col: "position",
    }
    # Add response columns generically (3P%, FG%, PTS, AST)
    for internal_name, csv_col in response_map.items():
        if csv_col in df.columns:
            rename_map[csv_col] = internal_name

    df.rename(columns=rename_map, inplace=True)

    # Create binary indicator: plays_with_curry (1) vs other teams (0)
    df["plays_with_curry"] = df["team"].isin(CURRY_TEAM_VALUES).astype(int)

    # Clean response columns: convert percentages like "40.5" or "0.405" to a 0–1 range
    for col in ["three_point_pct", "fg_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].dropna().max() > 1.5:
                df[col] = df[col] / 100.0

    # Drop rows with missing key predictor values
    required_for_model = ["minutes", "points", "age", "position"]
    df = df.dropna(subset=required_for_model)

    return df


def run_simple_analysis() -> None:
    """Run a very simple analysis and print results."""
    print("=" * 70)
    print("Curry Teammate Efficiency Analysis")
    print("=" * 70)

    df = load_and_prepare_data()
    print(f"\nLoaded {len(df)} players after cleaning.\n")

    # Metrics to analyze (only those present in df will be used)
    metrics = [
        ("three_point_pct", "3P%"),
        ("fg_pct", "FG%"),
        ("points", "PTS per game"),
        ("assists", "AST per game"),
    ]

    # One-hot encode position once (drop_first=True to avoid multicollinearity)
    df_dummies = pd.get_dummies(df, columns=["position"], drop_first=True)

    predictor_cols = [
        "plays_with_curry",  # categorical predictor (binary)
        "minutes",           # quantitative
        "points",            # quantitative
        "age",               # additional quantitative
    ]
    # Add position dummy columns as additional predictors
    position_cols = [c for c in df_dummies.columns if c.startswith("position_")]
    predictor_cols.extend(position_cols)

    for metric_col, metric_label in metrics:
        if metric_col not in df_dummies.columns:
            continue

        print("\n" + "=" * 70)
        print(f"Metric: {metric_label}")
        print("=" * 70)

        # Drop rows with missing response for this metric
        df_metric = df_dummies.dropna(subset=[metric_col]).copy()
        if df_metric.empty:
            print(f"\nSkipping {metric_label}: no non-missing values.")
            continue

        # 1) Descriptive comparison: with Curry vs other teams
        group_means = (
            df_metric.groupby("plays_with_curry")[metric_col]
            .mean()
            .rename(index={0: "Other teams", 1: "Plays with Curry"})
        )

        print(f"\nAverage {metric_label} by group:")
        for group, val in group_means.items():
            # Ensure val is a scalar (handle any unexpected pandas types)
            try:
                v = float(val)
            except TypeError:
                # If val is a Series or other type, take the first element
                v = float(getattr(val, "iloc", [val])[0])
            print(f"  {group:>15}: {v:.3f}")

        # 2) Simple multiple linear regression
        # Avoid using the response as a predictor (e.g. don't use 'points' to predict 'points')
        metric_predictors = [c for c in predictor_cols if c != metric_col]
        X = df_metric[metric_predictors].values
        y = df_metric[metric_col].values

        model = LinearRegression()
        model.fit(X, y)

        print(f"\nLinear Regression Results ({metric_label} as response):")
        print("-" * 70)

        def to_scalar(x):
            """Safely convert numpy scalar/array or Python scalar to float."""
            import numpy as np

            if isinstance(x, (float, int)):
                return float(x)
            if hasattr(x, "shape"):
                arr = np.asarray(x)
                return float(arr.ravel()[0])
            return float(x)

        for name, coef in zip(metric_predictors, model.coef_):
            c = to_scalar(coef)
            print(f"{name:>20}: {c:+.4f}")

        intercept = to_scalar(model.intercept_)
        print(f"{'intercept':>20}: {intercept:+.4f}")

        # Highlight the key quantity: effect of playing with Curry
        try:
            idx = metric_predictors.index("plays_with_curry")
            effect = to_scalar(model.coef_[idx])
            print("\nInterpretation:")
            print(
                f"  Holding minutes, points, age, and position constant, playing with Curry "
                f"is associated with a {effect:+.3f} change in {metric_label} (in 0–1 scale if percentage)."
            )
        except ValueError:
            print("\n(Note: 'plays_with_curry' predictor not found in model columns.)")


if __name__ == "__main__":
    run_simple_analysis()


