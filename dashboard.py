"""
Steph Curry Gravity Dashboard — 2015-16
Run: streamlit run dashboard.py
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, FancyArrow
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import fetch_all, CURRY_ID, SEASON
from analyze import zone_threat_gravity, def_dist_gravity


# ── Court drawing ─────────────────────────────────────────────────────────────
# NBA shot chart units: LOC_X/LOC_Y in tenths of feet from basket center.
# Court half = 470 units long, 500 units wide.

COURT_BG = "#0d1b2a"
LINE_C = "#7a93a8"


def _draw_court(ax):
    ax.set_facecolor(COURT_BG)
    kw = dict(fill=False, color=LINE_C, lw=1.5, zorder=1)
    # Paint
    ax.add_patch(mpatches.Rectangle((-80, -52.5), 160, 190, **kw))
    # Basket + backboard
    ax.add_patch(Circle((0, 0), 7.5, **kw))
    ax.plot([-30, 30], [-7.5, -7.5], color=LINE_C, lw=2, zorder=1)
    # Restricted area
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, **kw))
    # Free throw circle
    ax.add_patch(Arc((0, 190), 120, 120, theta1=0, theta2=180, **kw))
    ax.add_patch(Arc((0, 190), 120, 120, theta1=180, theta2=360,
                     color=LINE_C, lw=1.5, linestyle="dashed", fill=False, zorder=1))
    # 3-point line
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, **kw))
    ax.plot([-220, -220], [-52.5, 92.5], color=LINE_C, lw=1.5, zorder=1)
    ax.plot([220, 220], [-52.5, 92.5], color=LINE_C, lw=1.5, zorder=1)
    # Half-court
    ax.plot([-250, 250], [422.5, 422.5], color=LINE_C, lw=1, alpha=0.5, zorder=1)

    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 445)
    ax.set_aspect("equal")
    ax.axis("off")


# Zone patch regions (simplified polygons for fill)
def _zone_patches():
    """Returns dict of zone_name → matplotlib Patch for filling."""
    restricted = Circle((0, 0), 40, zorder=2)
    paint = mpatches.Rectangle((-80, -52.5), 160, 232.5, zorder=2)
    mid_range = Arc((0, 0), 475, 475, theta1=22, theta2=158, zorder=2)

    return {
        "Restricted Area":        (Circle((0, 0), 40), 2),
        "In The Paint (Non-RA)":  (mpatches.Rectangle((-80, 40), 160, 150), 2),
        "Mid-Range":              None,  # complex shape — skip fill, use scatter
        "Left Corner 3":          (mpatches.Rectangle((-260, -52.5), 40, 145), 2),
        "Right Corner 3":         (mpatches.Rectangle((220, -52.5), 40, 145), 2),
        "Above the Break 3":      None,  # arc region — use scatter
    }


ZONE_LABEL_POS = {
    "Restricted Area":       (0, 15),
    "In The Paint (Non-RA)": (0, 115),
    "Mid-Range":             (0, 215),
    "Left Corner 3":         (-237, 30),
    "Right Corner 3":        (237, 30),
    "Above the Break 3":     (0, 340),
}

ZONE_COLORS = {
    "Restricted Area":       "#264d73",
    "In The Paint (Non-RA)": "#1e5c40",
    "Mid-Range":             "#5c3d1e",
    "Left Corner 3":         "#5c1e4a",
    "Right Corner 3":        "#5c1e4a",
    "Above the Break 3":     "#4a1e5c",
}


def gravity_court_fig(ztg_df, curry_shots_df):
    """Court figure: Curry shot hexbin + zone gravity index annotations."""
    fig, ax = plt.subplots(figsize=(7, 7.5))
    fig.patch.set_facecolor(COURT_BG)

    # Shot hexbin (Curry)
    if curry_shots_df is not None and not curry_shots_df.empty:
        x = curry_shots_df["LOC_X"].values
        y = curry_shots_df["LOC_Y"].values
        mask = y < 420  # exclude backcourt
        hb = ax.hexbin(x[mask], y[mask], gridsize=22, cmap="YlOrRd",
                       mincnt=1, alpha=0.65, zorder=2, extent=(-260, 260, -60, 430))
        cbar = fig.colorbar(hb, ax=ax, fraction=0.022, pad=0.02, shrink=0.6)
        cbar.set_label("Shot frequency", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    _draw_court(ax)

    # Zone gravity labels
    zone_dict = dict(zip(ztg_df["zone"], ztg_df["zone_gravity_index"]))
    for zone, (lx, ly) in ZONE_LABEL_POS.items():
        g = zone_dict.get(zone)
        if g is None:
            continue
        color = "#00e676" if g >= 0 else "#ff5252"
        ax.text(lx, ly + 14, f"{g:+.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111cc",
                          edgecolor=color, linewidth=1.2))

    ax.set_title(
        "Curry Shot Heatmap  ·  Gravity Index per Zone  ·  2015-16\n"
        "Numbers = how much more of a scoring threat Curry is vs league avg",
        color="white", fontsize=9.5, pad=6,
    )
    fig.tight_layout()
    return fig


# ── Main dashboard ─────────────────────────────────────────────────────────────

def _data_ready():
    return (
        os.path.exists(os.path.join("data", "curry_shots.csv")) and
        os.path.exists(os.path.join("data", "zone_locations.csv")) and
        os.path.exists(os.path.join("data", "def_dist_overall.json")) and
        os.path.exists(os.path.join("data", "def_dist_catch_and_shoot.json"))
    )


def main():
    st.set_page_config(page_title="Curry Gravity", layout="wide", page_icon="🏀")
    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;font-size:2.2rem;'>🏀 Steph Curry — Gravity Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:15px;'>"
        f"2015-16 season &nbsp;·&nbsp; "
        "How much more defensive attention does Curry draw than an average player?</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Fetch gate ─────────────────────────────────────────────────────────────
    if not _data_ready():
        st.warning("No data yet. This downloads ~10 API calls from stats.nba.com (~30 sec).")
        if st.button("⬇️  Fetch 2015-16 Data"):
            with st.spinner("Downloading…"):
                try:
                    fetch_all()
                    st.success("Done! Reload the page.")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        st.stop()

    # ── Load & compute ─────────────────────────────────────────────────────────
    with st.spinner("Crunching numbers…"):
        try:
            ztg = zone_threat_gravity()
            g_all, per_player_all = def_dist_gravity(catch_and_shoot=False)
            g_cs,  per_player_cs  = def_dist_gravity(catch_and_shoot=True)
            curry_shots = pd.read_csv(os.path.join("data", "curry_shots.csv"))
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── KPI row ────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Overall Gravity",
        f"{g_all['gravity_ft']:+.2f} ft",
        help="League avg def dist minus Curry's avg def dist. Positive = more tightly guarded.",
    )
    k2.metric(
        "Off-Ball Gravity (C&S)",
        f"{g_cs['gravity_ft']:+.2f} ft",
        help="Same metric but only for catch-and-shoot shots — a proxy for off-ball tracking.",
    )
    k3.metric(
        "Curry tight-coverage %",
        f"{g_all['curry_tight_pct']:.1f}%",
        delta=f"{g_all['curry_tight_pct'] - g_all['league_avg_tight_pct']:+.1f}% vs league",
        delta_color="inverse",
        help="% of shots taken with defender within 4 feet.",
    )
    k4.metric(
        "Top gravity zone",
        ztg.iloc[0]["zone"] if not ztg.empty else "—",
        f"{ztg.iloc[0]['zone_gravity_index']:+.0f}% threat index" if not ztg.empty else "",
    )

    st.markdown("---")

    # ── Court map + zone gravity bar ───────────────────────────────────────────
    col_court, col_zone = st.columns([1, 1.05])

    with col_court:
        st.subheader("Shot Heatmap + Zone Gravity")
        st.caption("Numbers = Curry's zone threat vs league average (+% = bigger threat)")
        fig = gravity_court_fig(ztg, curry_shots)
        st.pyplot(fig, use_container_width=True)

    with col_zone:
        st.subheader("Zone Gravity Index")
        st.caption("How much more of a scoring threat is Curry vs the avg player in each zone?")

        fig_z = px.bar(
            ztg.sort_values("zone_gravity_index"),
            x="zone_gravity_index",
            y="zone",
            orientation="h",
            color="zone_gravity_index",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            text=ztg.sort_values("zone_gravity_index")["zone_gravity_index"].apply(
                lambda v: f"{v:+.0f}%"
            ),
            labels={"zone_gravity_index": "Gravity Index (%)", "zone": ""},
            template="plotly_dark",
        )
        fig_z.update_traces(textposition="outside")
        fig_z.update_layout(
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=5, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_z, use_container_width=True)

        with st.expander("What is Zone Gravity Index?"):
            st.markdown(
                "**Threat = (FGA share in zone) × FG% × point value (2 or 3)**\n\n"
                "Compares Curry's scoring threat per zone to the league average.\n"
                "A +200% index in 'Above the Break 3' means Curry is 3× more dangerous\n"
                "from there than the typical player → defenders *must* chase him there."
            )

    st.markdown("---")

    # ── Defender distance comparison ───────────────────────────────────────────
    st.subheader("Defender Distance — How Tightly Is Each Player Guarded?")
    tab1, tab2 = st.tabs(["All Shots", "Catch & Shoot (off-ball proxy)"])

    for tab, per_player, g in [(tab1, per_player_all, g_all), (tab2, per_player_cs, g_cs)]:
        with tab:
            pp = per_player.sort_values("avg_def_dist").copy()
            pp["is_curry"] = pp["PLAYER_ID"].astype(str) == str(CURRY_ID)
            pp["color"] = pp["is_curry"].map({True: "#FFD700", False: "#4a8fc4"})
            pp["label"] = pp["avg_def_dist"].apply(lambda v: f"{v:.2f}")

            fig_p = go.Figure(go.Bar(
                x=pp["PLAYER_NAME"],
                y=pp["avg_def_dist"],
                marker_color=pp["color"].tolist(),
                text=pp["label"],
                textposition="outside",
                hovertemplate="%{x}<br>Avg def dist: %{y:.2f} ft<extra></extra>",
            ))
            # League average line
            fig_p.add_hline(
                y=g["league_avg_def_dist"],
                line_dash="dash", line_color="#aaaaaa", line_width=1.5,
                annotation_text=f"League avg {g['league_avg_def_dist']:.2f} ft",
                annotation_font_color="#aaaaaa",
            )
            fig_p.update_layout(
                template="plotly_dark",
                height=380,
                yaxis_title="Avg closest defender (ft)  —  lower = more tightly guarded",
                xaxis_tickangle=-30,
                margin=dict(l=10, r=10, t=10, b=90),
                yaxis=dict(range=[0, pp["avg_def_dist"].max() * 1.22]),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("League avg def dist", f"{g['league_avg_def_dist']:.2f} ft")
            c2.metric("Curry avg def dist", f"{g['curry_avg_def_dist']:.2f} ft",
                      delta=f"{g['curry_avg_def_dist'] - g['league_avg_def_dist']:+.2f} ft",
                      delta_color="inverse")
            c3.metric("Curry tight coverage", f"{g['curry_tight_pct']:.1f}%",
                      delta=f"{g['curry_tight_pct'] - g['league_avg_tight_pct']:+.1f}% vs league",
                      delta_color="inverse")

    st.markdown("---")

    # ── Zone detail table ──────────────────────────────────────────────────────
    st.subheader("Zone Detail")
    disp = ztg.copy()
    disp["curry_fg_pct"] = (disp["curry_fg_pct"] * 100).round(1).astype(str) + "%"
    disp["curry_fga_freq"] = (disp["curry_fga_freq"] * 100).round(1).astype(str) + "%"
    disp["zone_gravity_index"] = disp["zone_gravity_index"].round(0).astype(int).astype(str) + "%"
    st.dataframe(
        disp.rename(columns={
            "zone": "Zone",
            "curry_fga": "Curry FGA",
            "curry_fga_freq": "Curry FGA%",
            "curry_fg_pct": "Curry FG%",
            "zone_gravity_index": "Gravity Index",
        })[["Zone", "Curry FGA", "Curry FGA%", "Curry FG%", "Gravity Index"]],
        use_container_width=True,
        hide_index=True,
    )

    # ── Def dist breakdown (tight % by range) ─────────────────────────────────
    st.markdown("---")
    st.subheader("Defender Distance Breakdown — Curry vs League")
    st.caption("How Curry's shot attempts distribute across defender-distance buckets vs the league average.")

    import json
    with open(os.path.join("data", "def_dist_overall.json")) as f:
        raw = json.load(f)

    dist_rows = []
    for rng, records in raw.items():
        df_r = pd.DataFrame(records)
        total_league_fga = df_r["FGA"].astype(float).sum()
        curry_fga = float(df_r[df_r["PLAYER_ID"].astype(str) == str(CURRY_ID)]["FGA"].values[0]) \
            if not df_r[df_r["PLAYER_ID"].astype(str) == str(CURRY_ID)].empty else 0
        dist_rows.append({
            "range": rng,
            "Curry FGA": curry_fga,
            "League total FGA": total_league_fga,
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df["Curry %"] = dist_df["Curry FGA"] / dist_df["Curry FGA"].sum() * 100
    dist_df["League avg %"] = dist_df["League total FGA"] / dist_df["League total FGA"].sum() * 100

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        name="Curry", x=dist_df["range"], y=dist_df["Curry %"],
        marker_color="#FFD700", text=dist_df["Curry %"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig_dist.add_trace(go.Bar(
        name="League avg", x=dist_df["range"], y=dist_df["League avg %"],
        marker_color="#4a8fc4", text=dist_df["League avg %"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
    ))
    fig_dist.update_layout(
        barmode="group", template="plotly_dark",
        height=360, yaxis_title="% of shots at this defender distance",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.caption(
        "A higher Curry % in the 'Very Tight' and 'Tight' buckets relative to league average "
        "shows defenders allocate more attention to him even when he's creating his shot."
    )


if __name__ == "__main__":
    main()
