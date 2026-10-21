"""
Steph Curry Gravity Dashboard — 2015-16
Run: streamlit run dashboard.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import fetch_all, CURRY_ID
from analyze import zone_threat_gravity, def_dist_gravity


# ── Court drawing ──────────────────────────────────────────────────────────────

COURT_BG = "#0d1b2a"
LINE_C = "#7a93a8"


def _draw_court(ax):
    ax.set_facecolor(COURT_BG)
    kw = dict(fill=False, color=LINE_C, lw=1.5, zorder=1)
    ax.add_patch(mpatches.Rectangle((-80, -52.5), 160, 190, **kw))
    ax.add_patch(Circle((0, 0), 7.5, **kw))
    ax.plot([-30, 30], [-7.5, -7.5], color=LINE_C, lw=2, zorder=1)
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, **kw))
    ax.add_patch(Arc((0, 190), 120, 120, theta1=0, theta2=180, **kw))
    ax.add_patch(Arc((0, 190), 120, 120, theta1=180, theta2=360,
                     color=LINE_C, lw=1.5, linestyle="dashed", fill=False, zorder=1))
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, **kw))
    ax.plot([-220, -220], [-52.5, 92.5], color=LINE_C, lw=1.5, zorder=1)
    ax.plot([220, 220], [-52.5, 92.5], color=LINE_C, lw=1.5, zorder=1)
    ax.plot([-250, 250], [422.5, 422.5], color=LINE_C, lw=1, alpha=0.5, zorder=1)
    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 445)
    ax.set_aspect("equal")
    ax.axis("off")


ZONE_LABEL_POS = {
    "Restricted Area":       (0, 15),
    "In The Paint (Non-RA)": (0, 115),
    "Mid-Range":             (0, 215),
    "Left Corner 3":         (-237, 30),
    "Right Corner 3":        (237, 30),
    "Above the Break 3":     (0, 340),
}


def gravity_court_fig(ztg_df, curry_shots_df):
    fig, ax = plt.subplots(figsize=(6, 6.5))
    fig.patch.set_facecolor(COURT_BG)

    if curry_shots_df is not None and not curry_shots_df.empty:
        x = curry_shots_df["LOC_X"].values
        y = curry_shots_df["LOC_Y"].values
        mask = y < 420
        hb = ax.hexbin(x[mask], y[mask], gridsize=20, cmap="YlOrRd",
                       mincnt=1, alpha=0.65, zorder=2, extent=(-260, 260, -60, 430))
        cbar = fig.colorbar(hb, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
        cbar.set_label("Shot volume", color="white", fontsize=7)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=6)

    _draw_court(ax)

    zone_dict = dict(zip(ztg_df["zone"], ztg_df["zone_gravity_index"]))
    for zone, (lx, ly) in ZONE_LABEL_POS.items():
        g = zone_dict.get(zone)
        if g is None:
            continue
        color = "#00e676" if g >= 0 else "#ff5252"
        ax.text(lx, ly + 14, f"{g:+.0f}%", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#111111cc",
                          edgecolor=color, linewidth=1.2))

    fig.tight_layout()
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _data_ready():
    return all(os.path.exists(os.path.join("data", f)) for f in [
        "curry_shots.csv", "zone_locations.csv",
        "def_dist_overall.json", "def_dist_catch_and_shoot.json",
    ])


def _callout(text):
    st.markdown(
        f"<div style='background:#1e3a5f;border-left:4px solid #4a9eff;"
        f"padding:10px 14px;border-radius:4px;color:#cde;font-size:14px;"
        f"margin-bottom:8px;'>{text}</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Curry Gravity", layout="centered", page_icon="🏀")

    st.markdown(
        "<h1 style='text-align:center;color:#FFD700;font-size:1.9rem;margin-bottom:4px;'>"
        "🏀 Steph Curry — Gravity Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:14px;margin-top:0;'>"
        "2015-16 season &nbsp;·&nbsp; NBA tracking data</p>",
        unsafe_allow_html=True,
    )

    # Plain-English intro
    with st.expander("What is 'gravity' in basketball? ▾", expanded=True):
        st.markdown(
            """
**Gravity** refers to how much a player forces defenders to pay attention to them,
even when they don't have the ball.

A player with high gravity is so dangerous as a shooter that defenders can't leave
them open — they have to chase him around the court constantly. This *opens up* lanes
and easy looks for teammates.

**How we measure it:**
1. **Zone Threat Index** — in each area of the court, how much more of a scoring
   threat is Curry compared to the average NBA player? A +975% index means defenders
   face a 10× bigger threat from him in that spot than from a typical player.
2. **Defender Distance** — how close is the nearest defender when Curry shoots?
   Lower distance = defenders are hugging him tighter = more gravity.
3. **Off-Ball Gravity** — on catch-and-shoot plays, Curry just received a pass and
   shot immediately. The defender distance here shows how closely he was being
   tracked *before he even touched the ball*.
            """
        )

    st.markdown("---")

    # ── Fetch gate ─────────────────────────────────────────────────────────────
    if not _data_ready():
        st.warning("No data yet. Click below to download from the NBA API (~30 sec).")
        if st.button("⬇️  Fetch 2015-16 Data"):
            with st.spinner("Downloading…"):
                try:
                    fetch_all()
                    st.success("Done! Reload the page.")
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
        st.stop()

    # ── Compute ────────────────────────────────────────────────────────────────
    with st.spinner("Calculating…"):
        try:
            ztg = zone_threat_gravity()
            g_all, per_player_all = def_dist_gravity(catch_and_shoot=False)
            g_cs,  per_player_cs  = def_dist_gravity(catch_and_shoot=True)
            curry_shots = pd.read_csv(os.path.join("data", "curry_shots.csv"))
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ── KPIs (2×2 grid so they don't squish on mobile) ─────────────────────────
    st.subheader("Key Numbers")
    _callout(
        "These four numbers summarize Curry's gravitational pull at a glance. "
        "Hover over each metric (desktop) or read the captions below for what they mean."
    )

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        st.metric(
            "Off-Ball Gravity",
            f"{g_cs['gravity_ft']:+.2f} ft",
        )
        st.caption(
            f"On catch-and-shoot plays, defenders stand **{g_cs['gravity_ft']:+.2f} ft** "
            f"closer to Curry than to the average player. "
            f"League avg: {g_cs['league_avg_def_dist']:.1f} ft · "
            f"Curry: {g_cs['curry_avg_def_dist']:.1f} ft."
        )

    with r1c2:
        st.metric(
            "Tight Coverage (off-ball)",
            f"{g_cs['curry_tight_pct']:.1f}%",
            delta=f"{g_cs['curry_tight_pct'] - g_cs['league_avg_tight_pct']:+.1f}% vs league",
            delta_color="inverse",
        )
        st.caption(
            f"**{g_cs['curry_tight_pct']:.1f}%** of Curry's catch-and-shoot attempts had "
            "a defender within 4 feet — arm's reach. The league average is "
            f"{g_cs['league_avg_tight_pct']:.1f}%. Defenders are already on him before he catches."
        )

    with r2c1:
        top = ztg.iloc[0]
        st.metric(
            "Highest Gravity Zone",
            top["zone"],
            f"{top['zone_gravity_index']:+.0f}% threat index",
        )
        st.caption(
            f"From **{top['zone']}**, Curry's expected scoring threat is "
            f"**{top['zone_gravity_index']:+.0f}%** higher than the average player. "
            "Defenders have no choice but to follow him there."
        )

    with r2c2:
        st.metric(
            "Overall Defender Distance",
            f"{g_all['curry_avg_def_dist']:.2f} ft",
            delta=f"{g_all['gravity_ft']:+.2f} ft vs league",
            delta_color="inverse",
        )
        st.caption(
            f"Across all his shots, the nearest defender is **{g_all['curry_avg_def_dist']:.2f} ft** "
            "away on average. This is slightly above league average because Curry takes "
            "many long 3s — see the off-ball number above for a cleaner read."
        )

    st.markdown("---")

    # ── Court map ──────────────────────────────────────────────────────────────
    st.subheader("Shot Heatmap + Zone Gravity")
    _callout(
        "The heatmap shows <b>where Curry took his 1,598 shots</b> (yellow/red = high volume). "
        "The <b>green numbers</b> show the gravity index for each zone — how much more of a "
        "scoring threat he is there vs an average player. Green = Curry is a bigger threat, "
        "red = closer to average.",
    )
    fig = gravity_court_fig(ztg, curry_shots)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # ── Zone gravity bar ───────────────────────────────────────────────────────
    st.subheader("Zone Gravity Index — by Court Area")
    _callout(
        "Each bar shows how much more dangerous Curry is vs the average player in that zone. "
        "<b>+975%</b> on the 3-point line means his scoring threat there is nearly <b>11× higher</b> "
        "than a typical player — defenders must sprint to cover him the moment he steps behind the arc."
    )

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
        hover_data={"zone_gravity_index": ":.0f"},
    )
    fig_z.update_traces(textposition="outside")
    fig_z.update_layout(
        coloraxis_showscale=False,
        height=360,
        margin=dict(l=5, r=70, t=10, b=10),
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_z, use_container_width=True)

    with st.expander("How is the Zone Gravity Index calculated?"):
        st.markdown(
            """
**Threat score** for a player in a zone = (share of their shots from that zone) × FG% × point value (2 or 3 pts)

This captures two things at once:
- **How often** they shoot from there (shot frequency)
- **How dangerous** those shots are (shooting percentage × points)

The gravity index = `(Curry's threat ÷ league average threat − 1) × 100%`

So +975% on the 3-point arc means Curry's scoring threat there is 10.75× the league average.
The average player rarely shoots 3s efficiently at high volume — Curry does both, making it
impossible for defenders to give him any daylight from that zone.
            """
        )

    st.markdown("---")

    # ── Defender distance tabs ─────────────────────────────────────────────────
    st.subheader("How Tightly Is Each Player Guarded?")
    _callout(
        "Lower bar = defender is standing closer = more defensive attention. "
        "The dashed line is the league average. <b>Gold bar = Curry.</b>"
    )

    tab1, tab2 = st.tabs(["All Shots", "Catch & Shoot (off-ball)"])

    for tab, per_player, g, description in [
        (tab1, per_player_all, g_all,
         "Average distance of the nearest defender across **all shot attempts**. "
         "Curry ranks near-average here because he takes many long 3s where even tight "
         "coverage means 5–6 ft — see the Catch & Shoot tab for the off-ball signal."),
        (tab2, per_player_cs, g_cs,
         "**Catch-and-shoot only**: the player received a pass and shot immediately (0 dribbles). "
         "The defender distance here reflects how closely they were tracked *before touching the ball* — "
         "this is the purest measure of off-ball gravity. Curry's bar being lower than most stars "
         "shows defenders are already chasing him before he even gets the ball."),
    ]:
        with tab:
            st.caption(description)

            pp = per_player.sort_values("avg_def_dist").copy()
            pp["is_curry"] = pp["PLAYER_ID"].astype(str) == str(CURRY_ID)
            pp["color"] = pp["is_curry"].map({True: "#FFD700", False: "#4a8fc4"})

            fig_p = go.Figure(go.Bar(
                x=pp["PLAYER_NAME"],
                y=pp["avg_def_dist"],
                marker_color=pp["color"].tolist(),
                text=pp["avg_def_dist"].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
                hovertemplate="%{x}<br>Avg defender distance: %{y:.2f} ft<extra></extra>",
            ))
            fig_p.add_hline(
                y=g["league_avg_def_dist"],
                line_dash="dash", line_color="#aaaaaa", line_width=1.5,
                annotation_text=f"League avg  {g['league_avg_def_dist']:.1f} ft",
                annotation_font_color="#aaaaaa", annotation_font_size=11,
            )
            fig_p.update_layout(
                template="plotly_dark",
                height=370,
                yaxis_title="Avg closest defender distance (ft)",
                xaxis_tickangle=-35,
                margin=dict(l=10, r=10, t=10, b=100),
                yaxis=dict(range=[0, pp["avg_def_dist"].max() * 1.25]),
                xaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_p, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("League avg", f"{g['league_avg_def_dist']:.2f} ft",
                      help="Weighted average across all players in the dataset.")
            m2.metric("Curry", f"{g['curry_avg_def_dist']:.2f} ft",
                      delta=f"{g['curry_avg_def_dist'] - g['league_avg_def_dist']:+.2f} ft vs league",
                      delta_color="inverse")
            m3.metric("Curry tight coverage",
                      f"{g['curry_tight_pct']:.1f}%",
                      delta=f"{g['curry_tight_pct'] - g['league_avg_tight_pct']:+.1f}% vs league",
                      delta_color="inverse",
                      help="% of shots with defender within 4 feet.")

    st.markdown("---")

    # ── Distance bucket breakdown ──────────────────────────────────────────────
    st.subheader("How Open Are Curry's Shots vs the League?")
    _callout(
        "Each group of bars shows what <b>percentage of shots</b> were taken at each defender distance. "
        "A higher Curry bar in 'Very Tight' (0–2 ft) or 'Tight' (2–4 ft) means more shots under heavy pressure. "
        "A higher Curry bar in 'Wide Open' (6+ ft) means he's getting more open looks — "
        "which is surprising given how hard defenders work to guard him."
    )

    with open(os.path.join("data", "def_dist_overall.json")) as f:
        raw = json.load(f)

    dist_rows = []
    for rng, records in raw.items():
        df_r = pd.DataFrame(records)
        total_league_fga = df_r["FGA"].astype(float).sum()
        curry_row = df_r[df_r["PLAYER_ID"].astype(str) == str(CURRY_ID)]
        curry_fga = float(curry_row["FGA"].values[0]) if not curry_row.empty else 0
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
        marker_color="#FFD700",
        text=dist_df["Curry %"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
        hovertemplate="%{x}<br>Curry: %{y:.1f}% of shots<extra></extra>",
    ))
    fig_dist.add_trace(go.Bar(
        name="League avg", x=dist_df["range"], y=dist_df["League avg %"],
        marker_color="#4a8fc4",
        text=dist_df["League avg %"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
        hovertemplate="%{x}<br>League avg: %{y:.1f}% of shots<extra></extra>",
    ))
    fig_dist.update_layout(
        barmode="group", template="plotly_dark",
        height=380,
        yaxis_title="% of total shot attempts",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.info(
        "**Key insight:** Curry gets a large share of 'Wide Open' looks (6+ ft) despite being "
        "one of the most closely tracked players in the league. This happens because he moves "
        "constantly off the ball — when defenders lose him for even a moment, he's already in "
        "a shooting stance behind the arc. The threat of that moment is gravity."
    )

    st.markdown("---")

    # ── Zone detail table ──────────────────────────────────────────────────────
    st.subheader("Zone Detail Table")
    st.caption(
        "FGA = field goal attempts. FGA% = share of Curry's total shots from this zone. "
        "FG% = shooting percentage. Gravity Index = how much bigger a threat Curry is here vs the league avg."
    )
    disp = ztg.copy()
    disp["curry_fg_pct"] = (disp["curry_fg_pct"] * 100).round(1).astype(str) + "%"
    disp["curry_fga_freq"] = (disp["curry_fga_freq"] * 100).round(1).astype(str) + "%"
    disp["zone_gravity_index"] = disp["zone_gravity_index"].round(0).astype(int).astype(str) + "%"
    st.dataframe(
        disp.rename(columns={
            "zone": "Zone",
            "curry_fga": "FGA",
            "curry_fga_freq": "FGA%",
            "curry_fg_pct": "FG%",
            "zone_gravity_index": "Gravity Index ↑",
        })[["Zone", "FGA", "FGA%", "FG%", "Gravity Index ↑"]],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
