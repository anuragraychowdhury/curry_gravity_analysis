"""
Fetch 2015-16 NBA data for Curry gravity analysis. Caches to data/.

Sources:
  1. ShotChartDetail(Curry) — exact shot locations for court visualization
  2. LeagueDashPlayerShotLocations — per-zone FGA/FG% for every player (1 call)
  3. LeagueDashPlayerPtShot × 4 defender-distance ranges — overall (4 calls)
  4. LeagueDashPlayerPtShot × 4 def ranges, Catch & Shoot — off-ball proxy (4 calls)
"""
import os
import json
import time
import pandas as pd
from nba_api.stats.endpoints import (
    ShotChartDetail,
    LeagueDashPlayerShotLocations,
    LeagueDashPlayerPtShot,
)

DATA_DIR = "data"
SEASON = "2015-16"
CURRY_ID = 201939

DEF_DIST_RANGES = [
    "0-2 Feet - Very Tight",
    "2-4 Feet - Tight",
    "4-6 Feet - Open",
    "6+ Feet - Wide Open",
]
# Mid-point feet value for each range (used for weighted avg calculation)
DEF_DIST_MIDPOINTS = {
    "0-2 Feet - Very Tight": 1.0,
    "2-4 Feet - Tight":      3.0,
    "4-6 Feet - Open":       5.0,
    "6+ Feet - Wide Open":   7.5,
}


def _sleep():
    time.sleep(1.2)


def fetch_curry_shotchart(force=False):
    path = os.path.join(DATA_DIR, "curry_shots.csv")
    if os.path.exists(path) and not force:
        return pd.read_csv(path)
    print("Fetching Curry shot chart…")
    _sleep()
    sc = ShotChartDetail(
        player_id=CURRY_ID, team_id=0,
        season_nullable=SEASON,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = sc.get_data_frames()[0]
    df.to_csv(path, index=False)
    print(f"  → {len(df)} shots")
    return df


def fetch_zone_locations(force=False):
    """Per-player per-zone FGA/FGM/FG% from LeagueDashPlayerShotLocations."""
    path = os.path.join(DATA_DIR, "zone_locations.csv")
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, header=[0, 1])
        return df
    print("Fetching zone shot locations (all players)…")
    _sleep()
    ep = LeagueDashPlayerShotLocations(
        season=SEASON,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        distance_range="By Zone",
        measure_type_simple="Base",
        last_n_games=0, month=0, opponent_team_id=0, period=0,
        plus_minus="N", pace_adjust="N", rank="N",
    )
    df = ep.get_data_frames()[0]
    df.to_csv(path)
    print(f"  → {len(df)} players")
    return df


def fetch_def_dist_data(general_range="", force=False):
    """
    Pull LeagueDashPlayerPtShot for all 4 defender-distance ranges.
    Returns dict: {range_label: DataFrame}.
    general_range: '' (overall) or 'Catch and Shoot'
    """
    tag = "overall" if not general_range else general_range.lower().replace(" ", "_")
    cache_path = os.path.join(DATA_DIR, f"def_dist_{tag}.json")

    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: pd.DataFrame(v) for k, v in raw.items()}

    label = f"  ({general_range})" if general_range else ""
    print(f"Fetching defender-distance data{label}…")
    result = {}
    for rng in DEF_DIST_RANGES:
        _sleep()
        kwargs = dict(
            season=SEASON,
            per_mode_simple="Totals",
            season_type_all_star="Regular Season",
            close_def_dist_range_nullable=rng,
        )
        if general_range:
            kwargs["general_range_nullable"] = general_range
        ep = LeagueDashPlayerPtShot(**kwargs)
        df = ep.get_data_frames()[0]
        result[rng] = df
        print(f"    {rng}: {len(df)} players")

    with open(cache_path, "w") as f:
        json.dump({k: v.to_dict(orient="records") for k, v in result.items()}, f)

    return result


def fetch_all(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_curry_shotchart(force=force)
    fetch_zone_locations(force=force)
    fetch_def_dist_data(general_range="", force=force)
    fetch_def_dist_data(general_range="Catch and Shoot", force=force)
    print("\nAll data fetched.")


if __name__ == "__main__":
    fetch_all()
