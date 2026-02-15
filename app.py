import math
import numpy as np
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="Location Intelligence Dashboard", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# -----------------------------
# Distance Functions
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def distance_m(lat1, lon1, lat2, lon2):
    return int(haversine_km(lat1, lon1, lat2, lon2) * 1000)

def density_per_km2(count, radius_m):
    area = math.pi * (radius_m/1000)**2
    return count / area if area > 0 else 0

# -----------------------------
# Demo Data Generator
# -----------------------------
def make_demo_points(center_lat, center_lon, category, radius_m, n):
    rng = np.random.default_rng(42)
    r_deg = (radius_m/1000)/111
    angles = rng.uniform(0, 2*np.pi, n)
    radii = r_deg * np.sqrt(rng.uniform(0, 1, n))

    lats = center_lat + radii*np.cos(angles)
    lons = center_lon + radii*np.sin(angles)/np.cos(np.deg2rad(center_lat))

    df = pd.DataFrame({
        "name": [f"{category.title()} #{i+1}" for i in range(n)],
        "category": category,
        "lat": lats.round(6),
        "lng": lons.round(6),
        "rating": np.clip(rng.normal(4.2,0.3,n),3,5).round(1),
        "review_count": np.clip(rng.normal(200,80,n).astype(int),5,1200),
        "is_competitor": rng.choice([True,False],size=n,p=[0.35,0.65])
    })

    df["distance_m"] = df.apply(
        lambda r: distance_m(center_lat,center_lon,r["lat"],r["lng"]), axis=1
    )

    score = (
        15
        + 45*((df["rating"]-3)/2)
        + 18*(np.log1p(df["review_count"])/np.log1p(1200))
        + 22*(1-df["distance_m"]/radius_m)
        - 7*df["is_competitor"].astype(int)
    )

    df["score"] = np.clip(score,0,100).round().astype(int)
    df["coverage_pct"] = (df["score"]/100).round(2)

    return df.sort_values("score",ascending=False).reset_index(drop=True)

# -----------------------------
# Overpass Block (Template)
# -----------------------------
def build_overpass_query(lat, lon, radius_m, category):
    tag_map = {
        "restaurant": 'node["amenity"="restaurant"]',
        "pharmacy": 'node["amenity"="pharmacy"]',
        "hospital": 'node["amenity"="hospital"]',
        "school": 'node["amenity"="school"]',
        "grocery": 'node["shop"="supermarket"]'
    }
    selector = tag_map.get(category,'node["amenity"]')

    return f"""
[out:json][timeout:25];
(
 {selector}(around:{radius_m},{lat},{lon});
);
out center tags;
"""

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="card">
<span class="badge">Analytics Dashboard</span>
<span class="badge">Location Intelligence</span>
<h1>🧭 Location Intelligence Dashboard</h1>
<div class="muted">
Executive-ready analytics for market saturation and opportunity scoring.
</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Parameters")

preset = st.sidebar.selectbox(
    "Area",
    ["Los Angeles (Downtown)", "Washington, DC", "New York (Midtown)", "Berlin (Mitte)"]
)

centers = {
    "Los Angeles (Downtown)": (34.052235, -118.243683),
    "Washington, DC": (38.907192, -77.036873),
    "New York (Midtown)": (40.754932, -73.984016),
    "Berlin (Mitte)": (52.520008, 13.404954),
}

center_lat, center_lon = centers[preset]

category = st.sidebar.selectbox("Category",
    ["restaurant","pharmacy","hospital","school","grocery"]
)

radius_m = st.sidebar.selectbox("Radius (meters)", [300,500,1000,1500,2000], index=2)
n_points = st.sidebar.slider("Number of results",10,120,50,5)
show_competitors = st.sidebar.checkbox("Include competitors",True)

st.sidebar.divider()
export_name = st.sidebar.text_input("CSV file name","location_intelligence_export.csv")

# -----------------------------
# Data
# -----------------------------
df = make_demo_points(center_lat,center_lon,category,radius_m,n_points)

if not show_competitors:
    df = df[df["is_competitor"]==False]

total = len(df)
avg_score = df["score"].mean()
avg_rating = df["rating"].mean()
density = density_per_km2(total,radius_m)
comp_share = df["is_competitor"].mean()
opp_index = (avg_score/100)*(1-comp_share)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📌 Overview","📋 Results","🧠 Method"])

with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Results",total)
    c2.metric("Avg Score",f"{avg_score:.1f}/100")
    c3.metric("Avg Rating",f"{avg_rating:.2f}⭐")
    c4.metric("Density",f"{density:.1f}/km²")
    c5.metric("Opportunity",f"{opp_index*100:.0f}%")

    st.map(df.rename(columns={"lat":"latitude","lng":"longitude"})[["latitude","longitude"]])

    st.subheader("Radius Sensitivity (same POIs, filtered by radius)")

    radii = [300, 500, 1000, 1500, 2000]

    # 1) Build ONE dataset at max radius using the same number of points as your dashboard
    max_r = max(radii)
    base = make_demo_points(center_lat, center_lon, category, max_r, n_points)

    if not show_competitors:
        base = base[base["is_competitor"] == False].reset_index(drop=True)

    rows = []
    for r in radii:
        sub = base[base["distance_m"] <= r].copy()

        total_r = len(sub)
        avg_score_r = float(sub["score"].mean()) if total_r else 0.0
        avg_rating_r = float(sub["rating"].mean()) if total_r else 0.0
        comp_share_r = float(sub["is_competitor"].mean()) if total_r else 0.0

        density_r = density_per_km2(total_r, r)
        opp_index_r = max(0.0, min(1.0, (avg_score_r / 100.0) * (1.0 - comp_share_r)))

        rows.append({
            "radius_m": r,
            "results_in_radius": total_r,
            "density_per_km2": round(density_r, 2),
            "avg_rating": round(avg_rating_r, 2),
            "avg_score": round(avg_score_r, 1),
            "competitor_share": round(comp_share_r, 3),
            "opportunity_%": round(opp_index_r * 100, 1),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab2:
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name=export_name,
                       mime="text/csv")

with tab3:
    st.markdown("""
This dashboard demonstrates a structured location intelligence pipeline.

In production:
- Data is fetched via Overpass API
- Distance calculations use Haversine
- Scores combine proximity, quality, demand proxy, and competitor signals
- Outputs are export-ready for planning and strategy
""")

st.caption("Demo data. Production version can connect to live backend.")
