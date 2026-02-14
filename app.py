import math
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
    competitor_flag = rng.choice([True, False], size=n, p=[0.35, 0.65])

    df = pd.DataFrame(
        {
            "name": names,
            "category": category,
            "lat": lats.round(6),
            "lng": lons.round(6),
            "rating": ratings.round(1),
            "review_count": reviews,
            "is_competitor": competitor_flag,
        }
    )

    df["distance_m"] = df.apply(
        lambda r: int(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000),
        axis=1,
    )

    # Realistic composite score (0..100)
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

    # Coverage proxy (0..1) — used only as a lightweight signal
    df["coverage_pct"] = np.clip((df["score"] / 100) * rng.uniform(0.7, 1.1, n), 0, 1).round(2)

    return df.sort_values(["score", "review_count"], ascending=False).reset_index(drop=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="card">
  <div>
    <span class="badge">Analytics Dashboard</span>
    <span class="badge">Location Intelligence</span>
    <span class="badge">KPIs • Map • Export</span>
  </div>
  <h1>🧭 Location Intelligence Dashboard</h1>
  <div class="muted">
    Executive-ready location analytics dashboard for market saturation analysis, competitor benchmarking, and site selection strategy.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Analysis Parameters")

city = st.sidebar.text_input("City / Area", "Los Angeles, CA")

preset = st.sidebar.selectbox(
    "Area",
    ["Los Angeles (Downtown)", "Washington, DC", "New York (Midtown)", "Berlin (Mitte)"],
)

category = st.sidebar.selectbox("Category", ["pharmacy", "restaurant", "hospital", "school", "grocery"])
radius_m = st.sidebar.selectbox("Radius (meters)", [300, 500, 1000, 1500, 2000], index=2)
n_points = st.sidebar.slider("Number of results", 10, 120, 45, step=5)

show_competitors = st.sidebar.checkbox("Include competitors", value=True)

st.sidebar.divider()
st.sidebar.subheader("Export")
export_name = st.sidebar.text_input("CSV file name", "location_intelligence_export.csv")

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

if not show_competitors:
    df = df[df["is_competitor"] == False].reset_index(drop=True)

total = len(df)
avg_score = float(df["score"].mean()) if total else 0.0
avg_rating = float(df["rating"].mean()) if total else 0.0
density = density_per_km2(total, radius_m)
def score_label(val: float) -> str:
    if val < 50:
        return "🔴 Weak"
    elif val < 70:
        return "🟠 Moderate"
    return "🟢 Strong"

def pct_label(val01: float) -> str:
    # val01 is 0..1
    pct = val01 * 100
    if pct < 30:
        return "🔴 Low"
    elif pct < 60:
        return "🟠 Medium"
    return "🟢 High"

comp_share = float(df["is_competitor"].mean()) if total else 0.0
opp_index = (avg_score / 100.0) * (1.0 - comp_share)
opp_index = max(0.0, min(1.0, opp_index))

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_results, tab_method = st.tabs(["📌 Overview", "📋 Results", "🧠 Method"])

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Results", total)
    c2.metric("Avg Score", f"{avg_score:.1f}/100", score_label(avg_score))
    c3.metric("Avg Rating", f"{avg_rating:.2f}⭐")
    c4.metric("Density", f"{density:.1f}/km²")
    c5.metric("Opportunity", f"{opp_index * 100:.0f}%", pct_label(opp_index))

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
- Toggle **Include competitors** to see impact on KPIs  
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
- **Density** is the number of results divided by the circular area within the selected radius.  
- **Opportunity** is a lightweight proxy combining average score and competitor share.  
"""
    )

    st.markdown("### Compliance note")
    st.markdown(
        """
This page includes **no external contact information** and is intended for portfolio demonstration.
"""
    )

st.write("")
st.caption("Note: Data shown is sample-formatted for demonstration. The dashboard can be connected to a live backend when needed.")
