"""
Streamlit app for Curry teammate efficiency analysis.

This app reuses the data and logic from `curry_effect.py` and presents:
- Group comparisons (plays with Curry vs other teams)
- Simple linear regression summaries
- Graphs to make results more interpretable
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Ensure project root is on sys.path so `analysis` package can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.curry_effect import load_and_prepare_data


def to_scalar(x):
    """Safely convert numpy scalar/array or Python scalar to float."""
    if isinstance(x, (float, int)):
        return float(x)
    arr = np.asarray(x)
    return float(arr.ravel()[0])


def main():
    st.set_page_config(page_title="Curry Teammate Impact Analysis", layout="wide")

    st.title("🏀 Curry Teammate Impact Analysis")
    st.markdown(
        """
        This app explores whether playing with **Stephen Curry** (on the Warriors)
        is associated with better performance compared to players on other teams.

        We compare:

        - **3P%** (three-point percentage)  
        - **FG%** (field goal percentage)  
        - **PTS per game**  
        - **AST per game**  

        while controlling for:

        - Minutes per game (MP)  
        - Age  
        - Position (guards / forwards / centers)  
        """
    )

    with st.spinner("Loading data..."):
        df = load_and_prepare_data()

    st.success(f"Loaded {len(df)} players from `nba_per_game_processed.csv`.")

    # Map display labels to internal column names
    metric_options = [
        ("3P% (three-point %)", "three_point_pct"),
        ("FG% (field goal %)", "fg_pct"),
        ("PTS per game", "points"),
        ("AST per game", "assists"),
    ]
    # Keep only metrics that actually exist in the loaded DataFrame
    metric_options = [opt for opt in metric_options if opt[1] in df.columns]

    st.sidebar.header("Analysis Settings")
    selected_metric = st.sidebar.selectbox(
        "Metric to analyze",
        metric_options,
        format_func=lambda x: x[0],
    )
    metric_label, metric_col = selected_metric

    # Prepare data for this metric
    df_metric = df.dropna(subset=[metric_col]).copy()

    if df_metric.empty:
        st.error(f"No data available for metric: {metric_label}")
        return

    # Descriptive comparison
    st.subheader(f"1. Group Comparison for {metric_label}")

    group_means = (
        df_metric.groupby("plays_with_curry")[metric_col]
        .mean()
        .rename(index={0: "Other teams", 1: "Plays with Curry"})
    )

    summary_df = pd.DataFrame(
        {
            "Group": group_means.index,
            metric_label: [to_scalar(v) for v in group_means.values],
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average values by group**")
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.markdown("**Visualization**")
        st.bar_chart(
            data=summary_df.set_index("Group"),
            use_container_width=True,
        )

    # Regression: does playing with Curry matter after controlling for other factors?
    st.subheader(f"2. Regression: Effect of Playing with Curry on {metric_label}")

    # One-hot encode position
    df_dummies = pd.get_dummies(df_metric, columns=["position"], drop_first=True)

    base_predictors = [
        "plays_with_curry",
        "minutes",
        "points",
        "age",
    ]
    # Avoid using the metric itself as a predictor when it's points
    metric_predictors = [p for p in base_predictors if p != metric_col]

    position_cols = [c for c in df_dummies.columns if c.startswith("position_")]
    metric_predictors.extend(position_cols)

    X = df_dummies[metric_predictors].values
    y = df_dummies[metric_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Align predictor names with actual coefficients produced by the model
    used_predictors = []
    used_coefs = []
    for name, coef in zip(metric_predictors, model.coef_):
        used_predictors.append(name)
        used_coefs.append(to_scalar(coef))

    intercept = to_scalar(model.intercept_)

    coef_df = pd.DataFrame(
        {
            "Predictor": used_predictors,
            "Coefficient": used_coefs,
        }
    )

    st.markdown("**Regression coefficients (simple linear model)**")
    st.dataframe(coef_df, use_container_width=True)

    st.markdown("**Coefficient visualization**")
    st.bar_chart(
        data=coef_df.set_index("Predictor"),
        use_container_width=True,
    )

    # Interpretation of plays_with_curry effect
    if "plays_with_curry" in used_predictors:
        effect = used_coefs[used_predictors.index("plays_with_curry")]
        st.markdown("### 3. Interpretation")

        if "per game" in metric_label.lower():
            unit = "points per game" if "PTS" in metric_label else "assists per game"
            st.write(
                f"- **Effect of playing with Curry**: "
                f"On average, holding minutes, age, and position constant, "
                f"players on the Warriors score **{effect:+.2f} {unit}** more "
                f"than comparable players on other teams."
            )
        else:
            st.write(
                f"- **Effect of playing with Curry**: "
                f"Holding minutes, points, age, and position constant, "
                f"playing with Curry is associated with a change of "
                f"**{effect:+.3f}** in {metric_label} (on a 0–1 scale if percentage)."
            )

        st.write(
            "- **Note**: This is a focused linear model on 2022–23 stats. The numbers summarize "
            "how strongly playing with Curry is associated with each metric after accounting for "
            "minutes, age, and position in this dataset."
        )

    # (Optional visualization like scatter plots can be added here later.
    #  For now we rely on the group bar chart and coefficient bar chart above
    #  to keep the app stable and easy to interpret.)


if __name__ == "__main__":
    main()


