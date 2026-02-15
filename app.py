import math
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Location Intelligence Dashboard", layout="wide")

# -----------------------------
# Light styling (safe + clean)
# -----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1, h2, h3 { margin-bottom: 0.3rem; }

.card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.03);
}
.muted { color: rgba(49, 51, 63, 0.65); font-size: 0.95rem; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.18);
  font-size: 0.85rem;
  margin-right: 8px;
}
.small { font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points (km)."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def density_per_km2(count: int, radius_m: int) -> float:
    area_km2 = math.pi * (radius_m / 1000.0) ** 2
    return (count / area_km2) if area_km2 > 0 else 0.0

def _split_keywords(text: str):
    parts = [p.strip() for p in (text or "").split(",")]
    return [p for p in parts if p]

def _keyword_regex(keywords):
    if not keywords:
        return None
    # Escape keywords, join into regex OR; word-boundary-ish matching
    escaped = [re.escape(k) for k in keywords if k]
    if not escaped:
        return None
    return re.compile("(" + "|".join(escaped) + ")", flags=re.IGNORECASE)

def make_demo_points(center_lat, center_lon, category, radius_m, n, seed=42):
    rng = np.random.default_rng(seed)

    # Spread points in a circle around the center
    r_deg = (radius_m / 1000.0) / 111.0  # approx degrees per km
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = r_deg * np.sqrt(rng.uniform(0, 1, n))

    lats = center_lat + radii * np.cos(angles)
    lons = center_lon + radii * np.sin(angles) / np.cos(np.deg2rad(center_lat))

    names = [f"{category.title()} #{i+1}" for i in range(n)]
    ratings = np.clip(rng.normal(4.2, 0.35, n), 3.0, 5.0)
    reviews = np.clip(rng.normal(180, 90, n).astype(int), 5, 1200)

    df = pd.DataFrame(
        {
            "name": names,
            "category": category,
            "lat": lats.round(6),
            "lng": lons.round(6),
            "rating": ratings.round(1),
            "review_count": reviews,
        }
    )

    df["distance_m"] = df.apply(
        lambda r: int(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000),
        axis=1,
    )

    # placeholder competitor flag (will be overwritten by real logic below)
    df["is_competitor"] = False

    return df

def ensure_score_column(
    dfr: pd.DataFrame,
    *,
    center_lat: float,
    center_lon: float,
    radius_m: int,
) -> pd.DataFrame:
    """
    Ensure dfr has a numeric 'score' column (0..100).
    If missing, build it from rating/review_count/distance_m/is_competitor.
    This prevents KeyError in Streamlit Cloud.
    """
    dfr = dfr.copy()

    # Ensure distance_m exists
    if "distance_m" not in dfr.columns:
        if {"lat", "lng"}.issubset(dfr.columns):
            dfr["distance_m"] = dfr.apply(
                lambda r: int(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000),
                axis=1,
            )
        else:
            dfr["distance_m"] = radius_m  # fallback

    # Ensure basic columns exist
    if "rating" not in dfr.columns:
        dfr["rating"] = 4.0
    if "review_count" not in dfr.columns:
        dfr["review_count"] = 50
    if "is_competitor" not in dfr.columns:
        dfr["is_competitor"] = False

    # Build score if missing
    if "score" not in dfr.columns:
        dist_norm = (dfr["distance_m"] / float(radius_m)).clip(0, 1)  # 0 close → 1 far
        rating_norm = ((dfr["rating"].astype(float) - 3.0) / 2.0).clip(0, 1)
        review_norm = (np.log1p(dfr["review_count"].astype(float)) / np.log1p(1200)).clip(0, 1)

        score = (
            15
            + 45 * rating_norm
            + 18 * review_norm
            + 22 * (1 - dist_norm)
            - 7 * dfr["is_competitor"].astype(int)
        )
        dfr["score"] = np.clip(np.round(score), 0, 100).astype(int)

    return dfr

def compute_score(df: pd.DataFrame, radius_m: int, seed: int = 42) -> pd.DataFrame:
    """Compute composite demo score (0..100) using rating/reviews/distance + competitor penalty."""
    if df.empty:
        df["score"] = []
        df["coverage_pct"] = []
        return df

    dist_norm = (df["distance_m"] / radius_m).clip(0, 1)                 # 0 close → 1 far
    rating_norm = ((df["rating"] - 3.0) / 2.0).clip(0, 1)               # 3..5 → 0..1
    review_norm = (np.log1p(df["review_count"]) / np.log1p(1200)).clip(0, 1)

    rng2 = np.random.default_rng(seed + 999)
    noise = rng2.normal(0, 4.0, len(df))  # add a little realism

    score = (
        15
        + 45 * rating_norm
        + 18 * review_norm
        + 22 * (1 - dist_norm)
        - 7 * df["is_competitor"].astype(int)
        + noise
    )
    df["score"] = np.clip(np.round(score), 0, 100).astype(int)
    df["coverage_pct"] = np.clip((df["score"] / 100), 0, 1).round(2)
    return df

def score_label(val: float) -> str:
    if val < 50:
        return "🔴 Weak"
    elif val < 70:
        return "🟠 Moderate"
    return "🟢 Strong"

def pct_label(val01: float) -> str:
    pct = val01 * 100
    if pct < 30:
        return "🔴 Low"
    elif pct < 60:
        return "🟠 Medium"
    return "🟢 High"
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def band_label_01(val01: float, low: float = 0.33, mid: float = 0.66) -> str:
    """
    Generic label for 0..1 signals.
    """
    if val01 < low:
        return "🟢 Low"
    elif val01 < mid:
        return "🟠 Medium"
    return "🔴 High"

def competitive_pressure_index(density: float, comp_share: float, density_ref: float = 30.0) -> float:
    """
    Competitive Pressure Index (0..1):
    - density_ref is a "high density" reference point (per km²).
    - comp_share is 0..1 (share of listings flagged as competitors).

    Pressure rises with both density and competitor share.
    """
    dens01 = clamp01(density / density_ref)          # 0..1
    comp01 = clamp01(comp_share)                      # 0..1
    pressure01 = 0.60 * dens01 + 0.40 * comp01        # weighted blend
    return clamp01(pressure01)

def entry_risk_index(opp_index: float, pressure01: float) -> float:
    """
    Entry Risk Indicator (0..1):
    - higher risk when opportunity is low and competitive pressure is high
    """
    opp01 = clamp01(opp_index)
    pres01 = clamp01(pressure01)
    risk01 = 0.55 * (1.0 - opp01) + 0.45 * pres01
    return clamp01(risk01)

def opportunity_recommendation(
    opp_index: float,
    density: float,
    comp_share: float,
    radius_m: int
):
    """
    Executive-grade market recommendation logic.
    Context-aware: density interpreted relative to radius.
    """

    pct = opp_index * 100
    comp_pct = comp_share * 100

    # Context-aware density interpretation
    density_equivalent = density * (radius_m / 1000)

    if density_equivalent > 25:
        saturation_label = "highly saturated"
    elif density_equivalent > 12:
        saturation_label = "moderately saturated"
    else:
        saturation_label = "structurally underserved"

    # Opportunity interpretation
    if pct < 30:
        return {
            "headline": "Constrained Market Entry Conditions",
            "text": (
                f"The micro-market appears {saturation_label} with elevated competitive intensity "
                f"({comp_pct:.0f}% competitor share). Risk-adjusted entry attractiveness is limited."
            ),
            "action": (
                "Reassess catchment definition, evaluate adjacent underserved pockets, "
                "or refine differentiation strategy before capital commitment."
            ),
            "color": "#ffe5e5"
        }

    elif pct < 60:
        return {
            "headline": "Selective Opportunity",
            "text": (
                f"The area shows {saturation_label} conditions with moderate competitive presence "
                f"({comp_pct:.0f}% competitor share). Performance will depend on micro-location quality "
                "and brand positioning."
            ),
            "action": (
                "Shortlist high-footfall corners, test proximity to anchors, "
                "and conduct rental benchmarking prior to decision."
            ),
            "color": "#fff4e0"
        }

    else:
        return {
            "headline": "Favorable Market Entry Conditions",
            "text": (
                f"The catchment appears {saturation_label} with manageable competitive pressure "
                f"({comp_pct:.0f}% competitor share). Market signals support expansion or new site feasibility."
            ),
            "action": (
                "Proceed with due diligence: validate lease economics, "
                "customer flow patterns, and competitive differentiation."
            ),
            "color": "#e6f4ea"
        }

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Analysis Parameters")

city = st.sidebar.text_input("City / Area", "Los Angeles, CA", key="city")

preset = st.sidebar.selectbox(
    "Area",
    ["Los Angeles (Downtown)", "Washington, DC", "New York (Midtown)", "Berlin (Mitte)"],
    key="preset",
)

category = st.sidebar.selectbox(
    "Category",
    ["pharmacy", "restaurant", "hospital", "school", "grocery"],
    key="category",
)

radius_m = st.sidebar.selectbox(
    "Radius (meters)",
    [300, 500, 1000, 1500, 2000],
    index=2,
    key="radius_m",
)

n_points = st.sidebar.slider(
    "Number of results",
    10, 120, 45, step=5,
    key="n_points",
)

st.sidebar.divider()
st.sidebar.subheader("Competitor Definition (optional)")

competitor_keywords_text = st.sidebar.text_input(
    "Competitor keywords (comma-separated)",
    value="",
    help="Example: CVS, Walgreens, Starbucks. Leave blank for demo competitor labeling.",
    key="competitor_keywords",
)

show_competitors = st.sidebar.checkbox("Include competitors", value=True, key="show_competitors")

st.sidebar.divider()
st.sidebar.subheader("Export")

export_name = st.sidebar.text_input(
    "CSV file name",
    "location_intelligence_export.csv",
    key="export_name",
)

centers = {
    "Los Angeles (Downtown)": (34.052235, -118.243683),
    "Washington, DC": (38.907192, -77.036873),
    "New York (Midtown)": (40.754932, -73.984016),
    "Berlin (Mitte)": (52.520008, 13.404954),
}
center_lat, center_lon = centers[preset]

seed = 42  # stable demo runs

# -----------------------------
# Data
# -----------------------------
df = make_demo_points(center_lat, center_lon, category, radius_m, n_points, seed=seed)

# Competitor labeling: keyword-based if provided, otherwise demo distribution
kw = _split_keywords(competitor_keywords_text)
rx = _keyword_regex(kw)

if rx is not None:
    df["is_competitor"] = df["name"].astype(str).apply(lambda s: bool(rx.search(s)))
else:
    rng = np.random.default_rng(seed + 123)
    df["is_competitor"] = rng.choice([True, False], size=len(df), p=[0.35, 0.65])

# Apply competitor filter (keeps your existing view/behavior)
if not show_competitors:
    df = df[df["is_competitor"] == False].reset_index(drop=True)

# Compute score AFTER competitor assignment (so penalty is meaningful)
df = compute_score(df, radius_m=radius_m, seed=seed)

# Sort like your original view
df = df.sort_values(["score", "review_count"], ascending=False).reset_index(drop=True)

total = len(df)
avg_score = float(df["score"].mean()) if total else 0.0
avg_rating = float(df["rating"].mean()) if total else 0.0
density = density_per_km2(total, radius_m)

comp_share = float(df["is_competitor"].mean()) if total else 0.0
opp_index = (avg_score / 100.0) * (1.0 - comp_share)
opp_index = max(0.0, min(1.0, opp_index))
# --- NEW: Competitive pressure + entry risk
pressure01 = competitive_pressure_index(density=density, comp_share=comp_share, density_ref=30.0)
risk01 = entry_risk_index(opp_index=opp_index, pressure01=pressure01)

pressure_score = int(round(pressure01 * 100))
risk_score = int(round(risk01 * 100))

pressure_label = band_label_01(pressure01)  # Low/Medium/High (pressure)
risk_label = band_label_01(risk01)          # Low/Medium/High (risk)

# -----------------------------
# Header (keep existing view; improved positioning)
# -----------------------------
st.markdown(
    f"""
<div class="card">
  <div>
    <span class="badge">Decision Support</span>
    <span class="badge">Site Selection</span>
    <span class="badge">Market Saturation</span>
    <span class="badge">Competitor Pressure</span>
  </div>
  <h1>🧭 Location Intelligence Dashboard</h1>
  <div class="muted">
    Executive-ready micro-market screening for <b>site selection</b>, <b>retail strategy</b>, and <b>feasibility briefs</b>.
    <span class="small">Anchor: <b>{preset}</b> • Category: <b>{category}</b> • Radius: <b>{radius_m}m</b></span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_results, tab_method = st.tabs(["📌 Overview", "📋 Results", "🧠 Method"])

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Results", total)
    c2.metric("Avg Score", f"{avg_score:.1f}/100", score_label(avg_score))
    c3.metric("Avg Rating", f"{avg_rating:.2f}⭐")
    c4.metric("Density (/km²)", f"{density:.1f}")

    c5.metric("Opportunity", f"{opp_index * 100:.0f}%", pct_label(opp_index))
    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.metric("Competitive Pressure", f"{pressure_score}/100", pressure_label)
    k2.metric("Entry Risk", f"{risk_score}/100", risk_label)
    k3.metric("Competitor Share", f"{comp_share * 100:.0f}%")

    st.write("")
    st.markdown("### Multi-radius snapshot")

    snapshot_radii = [300, 500, 1000, 1500, 2000]
    rows = []

    for rr in snapshot_radii:
        dfr = make_demo_points(center_lat, center_lon, category, rr, n_points, seed=seed)

        if not show_competitors:
            dfr = dfr[dfr["is_competitor"] == False].reset_index(drop=True)

        # ✅ FIX: guarantee 'score' exists
        dfr = ensure_score_column(dfr, center_lat=center_lat, center_lon=center_lon, radius_m=rr)

        tot_r = len(dfr)
        avg_score_r = float(dfr["score"].mean()) if tot_r else 0.0
        avg_rating_r = float(dfr["rating"].mean()) if tot_r else 0.0
        dens_r = density_per_km2(tot_r, rr)
        comp_share_r = float(dfr["is_competitor"].mean()) if tot_r else 0.0

        opp_r = clamp01((avg_score_r / 100.0) * (1.0 - comp_share_r))
        pres_r = competitive_pressure_index(density=dens_r, comp_share=comp_share_r, density_ref=30.0)
        risk_r = entry_risk_index(opp_index=opp_r, pressure01=pres_r)

        rows.append({
            "Radius (m)": rr,
            "Results": tot_r,
            "Avg Score": round(avg_score_r, 1),
            "Avg Rating": round(avg_rating_r, 2),
            "Density (/km²)": round(dens_r, 1),
            "Competitor %": int(round(comp_share_r * 100)),
            "Opportunity %": int(round(opp_r * 100)),
            "Pressure": int(round(pres_r * 100)),
            "Risk": int(round(risk_r * 100)),
        })

    snap = pd.DataFrame(rows)
    st.dataframe(snap, use_container_width=True, height=240)
    st.caption("Tip: Compare how opportunity/pressure/risk shifts as you widen the catchment radius.")

    st.write("")
    st.markdown("### Executive Insight")

    rec = opportunity_recommendation(
        opp_index=opp_index,
        density=density,
        comp_share=comp_share,
        radius_m=radius_m
    )

    st.markdown(
        f"""
<div style="
    border-radius:16px;
    padding:16px;
    background:{rec['color']};
    border:1px solid rgba(0,0,0,0.05);
">
  <div style="font-weight:800; font-size:1.05rem; margin-bottom:6px;">
    {rec['headline']}
  </div>

  <div style="line-height:1.6; margin-bottom:10px;">
    {rec['text']}
  </div>

  <div class="muted small" style="line-height:1.6; margin-bottom:6px;">
    <b>Recommended next step:</b> {rec['action']}
  </div>

  <div class="muted small" style="line-height:1.6;">
Basis: Opportunity {opp_index * 100:.0f}% • Avg score {avg_score:.1f} • Competitor share {comp_share * 100:.0f}% • Density {density:.1f}/km² • Pressure {pressure_score}/100 • Risk {risk_score}/100

  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### Business value")
        st.markdown(
            """
- **Market saturation signal:** density + competitor share to gauge crowded vs underserved areas  
- **Benchmarking:** compare categories, radii, and zones consistently  
- **Decision support:** export-ready reporting for **site selection**, **retail strategy**, and **investment memos**  
- **Stakeholder-friendly:** KPIs + map + downloadable dataset for quick review
"""
        )

        st.markdown("### Quick actions")
        st.markdown(
            """
- Change **Category** and **Radius** in the sidebar  
- (Optional) add **Competitor keywords** to define competition  
- Toggle **Include competitors** to see KPI impact  
- Use **Download CSV** for reporting-ready output
"""
        )

    with right:
        st.markdown("### Map preview")
        map_df = df[["lat", "lng"]].rename(columns={"lat": "latitude", "lng": "longitude"})
        st.map(map_df, zoom=12)
        st.markdown(
            f"<div class='muted small'>Anchor: {preset} • Radius: {radius_m}m • City label: {city}</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Score distribution")
    hist = df[["score"]].copy()
    hist["bucket"] = (hist["score"] // 10) * 10
    dist = hist.groupby("bucket").size().reset_index(name="count").sort_values("bucket").set_index("bucket")
    st.bar_chart(dist)
if competitor_keywords_text.strip():
    st.markdown(
        f"<div class='muted small'>Competitors defined by keywords: <b>{competitor_keywords_text}</b></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='muted small'>Competitors are demo-labeled (add keywords for real competitor definition).</div>",
        unsafe_allow_html=True,
    )

with tab_results:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Results table")
        view_cols = ["name", "category", "rating", "review_count", "distance_m", "score", "is_competitor"]
        st.dataframe(df[view_cols], use_container_width=True, height=520)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=export_name,
            mime="text/csv",
        )

    with right:
        st.subheader("Top performers")
        top = df.sort_values(["score", "rating", "review_count"], ascending=False).head(10)
        st.dataframe(
            top[["name", "rating", "review_count", "distance_m", "score", "is_competitor"]],
            use_container_width=True,
            height=350,
        )

        st.subheader("Competitor share")
        comp = pd.DataFrame(
            {
                "segment": ["Competitors", "Non-competitors"],
                "count": [int(df["is_competitor"].sum()), int((~df["is_competitor"]).sum())],
            }
        ).set_index("segment")
        st.bar_chart(comp)

with tab_method:
    st.markdown("### What this dashboard represents")
    st.markdown(
        """
This dashboard illustrates the structure and reporting format of an automated location analytics pipeline.  
Data shown here is **sample-formatted for demonstration purposes**.

In production, the same dashboard can be connected to a live pipeline that:
- Collects POIs from selected sources
- Cleans & deduplicates entities
- Applies scoring rules (distance, quality signals, category relevance, competitor labeling)
- Outputs structured datasets suitable for analytics and decision-making
"""
    )

    st.markdown("### Interpretation notes")
    st.markdown(
        """
- **Score** is a composite signal from proximity, quality (rating), demand proxy (reviews), and competitor labeling.  
- **Density** is results divided by the circular area within the selected radius.  
- **Opportunity** combines average score and competitor share as a lightweight screening proxy.  
"""
    )

    st.markdown("### Competitor definition")
    st.markdown(
        """
- If you provide **Competitor keywords**, entries whose **name matches** those keywords are labeled as competitors.  
- If you leave it blank, competitors are **demo-labeled** to simulate market pressure.  
"""
    )

st.caption("Note: Data shown is sample-formatted for demonstration. The dashboard can be connected to a live backend when needed.")
