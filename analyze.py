"""
Curry gravity analysis.

Two complementary gravity metrics:

1. ZONE THREAT GRAVITY
   Per court zone: compare Curry's scoring threat (FGA_frequency × FG%) to the
   league average. Where Curry's threat far exceeds average, defenders *must*
   collapse on him — that's gravitational pull.
   Zone threat index = (Curry threat / League avg threat) − 1
   Positive = Curry is a bigger threat here than average → forces defensive attention.

2. DEFENDER DISTANCE GRAVITY
   From LeagueDashPlayerPtShot (4 calls per distance bucket):
   Compute each player's weighted-average closest-defender distance.
   Gravity = league_avg_def_dist − curry_avg_def_dist
   Positive = Curry gets closer defenders than average → more attention drawn.
   Run separately for all shots and for catch-and-shoot (off-ball proxy).
"""
import os
import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
CURRY_ID = 201939

ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]

DEF_DIST_RANGES = [
    "0-2 Feet - Very Tight",
    "2-4 Feet - Tight",
    "4-6 Feet - Open",
    "6+ Feet - Wide Open",
]
DEF_DIST_MIDPOINTS = {
    "0-2 Feet - Very Tight": 1.0,
    "2-4 Feet - Tight":      3.0,
    "4-6 Feet - Open":       5.0,
    "6+ Feet - Wide Open":   7.5,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path} — run: python fetch_data.py")
    return path


def _load_def_dist(tag):
    path = _require(os.path.join(DATA_DIR, f"def_dist_{tag}.json"))
    with open(path) as f:
        raw = json.load(f)
    return {k: pd.DataFrame(v) for k, v in raw.items()}


# ── Zone threat analysis ──────────────────────────────────────────────────────

def zone_threat_gravity():
    """
    Returns DataFrame with columns:
      zone, curry_fga, curry_fga_freq, curry_fg_pct, curry_threat,
      league_avg_threat, zone_gravity_index
    """
    df = pd.read_csv(_require(os.path.join(DATA_DIR, "zone_locations.csv")), header=[0, 1])

    # Flatten multi-level columns
    df.columns = [
        f"{zone}_{stat}" if zone.strip() else stat
        for zone, stat in df.columns
    ]
    df.columns = df.columns.str.strip()

    # Identify player id / name columns
    id_col = [c for c in df.columns if "PLAYER_ID" in c][0]
    name_col = [c for c in df.columns if "PLAYER_NAME" in c][0]

    rows = []
    for zone in ZONES:
        fga_col = f"{zone}_FGA"
        fgm_col = f"{zone}_FGM"
        pct_col = f"{zone}_FG_PCT"

        if fga_col not in df.columns:
            continue

        sub = df[[id_col, name_col, fga_col, fgm_col, pct_col]].copy()
        sub.columns = ["PLAYER_ID", "PLAYER_NAME", "FGA", "FGM", "FG_PCT"]
        sub["FGA"] = pd.to_numeric(sub["FGA"], errors="coerce").fillna(0)
        sub["FGM"] = pd.to_numeric(sub["FGM"], errors="coerce").fillna(0)
        sub["FG_PCT"] = pd.to_numeric(sub["FG_PCT"], errors="coerce")

        total_fga = sub["FGA"].sum()
        if total_fga == 0:
            continue

        # FGA frequency: fraction of all shots taken in this zone
        total = max(sub["FGA"].sum(), 1)
        sub["fga_freq"] = sub["FGA"] / total

        # Pts-per-FGA threat (3pt zones score 3, others score 2)
        pts = 3 if "3" in zone or "Break 3" in zone else 2
        sub["threat"] = sub["fga_freq"] * sub["FG_PCT"].fillna(0) * pts

        league_avg_threat = sub["threat"].mean()
        curry_row = sub[sub["PLAYER_ID"].astype(str) == str(CURRY_ID)]
        if curry_row.empty:
            continue

        c = curry_row.iloc[0]
        rows.append({
            "zone": zone,
            "curry_fga": int(c["FGA"]),
            "curry_fga_freq": float(c["fga_freq"]),
            "curry_fg_pct": float(c["FG_PCT"]) if not pd.isna(c["FG_PCT"]) else 0,
            "curry_threat": float(c["threat"]),
            "league_avg_threat": league_avg_threat,
            "zone_gravity_index": (
                (float(c["threat"]) / league_avg_threat - 1) * 100
                if league_avg_threat > 0 else 0
            ),
            "pts_value": pts,
        })

    return pd.DataFrame(rows).sort_values("zone_gravity_index", ascending=False).reset_index(drop=True)


# ── Defender distance analysis ────────────────────────────────────────────────

def _weighted_def_dist(dist_data: dict) -> pd.DataFrame:
    """
    From 4 DataFrames (one per distance range), compute each player's
    weighted-average closest-defender distance.
    """
    frames = []
    for rng, df in dist_data.items():
        mid = DEF_DIST_MIDPOINTS[rng]
        sub = df[["PLAYER_ID", "PLAYER_NAME", "FGA"]].copy()
        sub["FGA"] = pd.to_numeric(sub["FGA"], errors="coerce").fillna(0)
        sub["weighted_ft"] = sub["FGA"] * mid
        sub["range"] = rng
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    agg = (
        combined.groupby(["PLAYER_ID", "PLAYER_NAME"])
        .apply(lambda g: pd.Series({
            "total_fga": g["FGA"].sum(),
            "avg_def_dist": (g["weighted_ft"].sum() / g["FGA"].sum()
                             if g["FGA"].sum() > 0 else np.nan),
            "pct_tight": (g.loc[g["range"].isin(["0-2 Feet - Very Tight", "2-4 Feet - Tight"]),
                                 "FGA"].sum() / g["FGA"].sum()
                          if g["FGA"].sum() > 0 else np.nan),
        }), include_groups=False)
        .reset_index()
    )
    return agg.dropna(subset=["avg_def_dist"])


def def_dist_gravity(catch_and_shoot=False):
    """
    Returns (gravity_df, per_player_df).
    gravity_df: overall league stats + Curry comparison
    per_player_df: one row per player with avg_def_dist, pct_tight
    """
    tag = "catch_and_shoot" if catch_and_shoot else "overall"
    dist_data = _load_def_dist(tag)
    per_player = _weighted_def_dist(dist_data)

    league_avg_dist = np.average(per_player["avg_def_dist"], weights=per_player["total_fga"])
    league_avg_tight = np.average(per_player["pct_tight"], weights=per_player["total_fga"])

    curry = per_player[per_player["PLAYER_ID"].astype(str) == str(CURRY_ID)]
    if curry.empty:
        raise ValueError("Curry not found in defender distance data.")

    c = curry.iloc[0]
    gravity = {
        "league_avg_def_dist": round(float(league_avg_dist), 3),
        "curry_avg_def_dist":  round(float(c["avg_def_dist"]), 3),
        "gravity_ft":          round(float(league_avg_dist - c["avg_def_dist"]), 3),
        "league_avg_tight_pct": round(float(league_avg_tight) * 100, 1),
        "curry_tight_pct":      round(float(c["pct_tight"]) * 100, 1),
        "catch_and_shoot":      catch_and_shoot,
    }
    return gravity, per_player


if __name__ == "__main__":
    print("=== Zone Threat Gravity ===")
    ztg = zone_threat_gravity()
    pd.set_option("display.float_format", "{:.3f}".format)
    print(ztg[["zone", "curry_fga", "curry_fg_pct", "curry_threat",
               "league_avg_threat", "zone_gravity_index"]].to_string(index=False))

    print("\n=== Defender Distance Gravity (All Shots) ===")
    g_all, _ = def_dist_gravity(catch_and_shoot=False)
    for k, v in g_all.items():
        print(f"  {k}: {v}")

    print("\n=== Defender Distance Gravity (Catch & Shoot — off-ball) ===")
    g_cs, _ = def_dist_gravity(catch_and_shoot=True)
    for k, v in g_cs.items():
        print(f"  {k}: {v}")
