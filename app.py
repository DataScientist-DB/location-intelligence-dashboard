import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Location Intelligence Dashboard", layout="wide")

def haversine_km(lat1, lon1, lat2, lon2) -> float:
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
    r_deg = (radius_m / 1000.0) / 111.0
    angles = rng.uniform(0, 2*np.pi, n)
    radii = r_deg * np.sqrt(rng.uniform(0, 1, n))

    lats = center_lat + radii * np.cos(angles)
    lons = center_lon + radii * np.sin(angles) / np.cos(np.deg2rad(center_lat))

    names = [f"{category.title()} #{i+1}" for i in range(n)]
    ratings = np.clip(rng.normal(4.2, 0.35, n), 3.0, 5.0)
    reviews = np.clip(rng.normal(180, 90, n).astype(int), 5, 1200)
    competitor_flag = rng.choice([True, False], size=n, p=[0.35, 0.65])

    df = pd.DataFrame({
        "name": names,
        "category": category,
        "lat": lats.round(6),
        "lng": lons.round(6),
        "rating": ratings.round(1),
        "review_count": reviews,
        "is_competitor": competitor_flag,
    })

    df["distance_m"] = df.apply(
        lambda r: int(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000),
        axis=1
    )

    score = (
        (df["rating"] * 18)
        + np.log1p(df["review_count"]) * 10
        + (1 - (df["distance_m"] / radius_m).clip(0, 1)) * 25
        - df["is_competitor"].astype(int) * 6
    )
    df["score"] = np.clip(score.round(0), 0, 100).astype(int)
    df["coverage_pct"] = np.clip((df["score"] / 100) * rng.uniform(0.7, 1.1, n), 0, 1).round(2)

    return df.sort_values(["score", "review_count"], ascending=False).reset_index(drop=True)

st.title("🧭 Location Intelligence Dashboard")
st.caption("Portfolio demo dashboard showing sample outputs from an automated location data pipeline.")

st.sidebar.header("Scenario")
city = st.sidebar.text_input("City / Area", "Los Angeles, CA")

preset = st.sidebar.selectbox(
    "Demo scenario",
    ["LA Downtown", "Washington, DC", "New York Midtown", "Berlin Mitte"]
)
category = st.sidebar.selectbox("Category", ["pharmacy", "restaurant", "hospital", "school", "grocery"])
radius_m = st.sidebar.selectbox("Radius (meters)", [300, 500, 1000, 1500, 2000], index=2)
n_points = st.sidebar.slider("Number of results", 10, 120, 45, step=5)
seed = st.sidebar.number_input("Random seed (demo)", value=42, step=1)
show_competitors = st.sidebar.checkbox("Include competitors", value=True)
export_name = st.sidebar.text_input("CSV file name", "location_intelligence_demo.csv")

centers = {
    "LA Downtown": (34.052235, -118.243683),
    "Washington, DC": (38.907192, -77.036873),
    "New York Midtown": (40.754932, -73.984016),
    "Berlin Mitte": (52.520008, 13.404954),
}
center_lat, center_lon = centers[preset]

df = make_demo_points(center_lat, center_lon, category, radius_m, n_points, seed=int(seed))
if not show_competitors:
    df = df[df["is_competitor"] == False].reset_index(drop=True)

total = len(df)
avg_score = float(df["score"].mean()) if total else 0.0
avg_rating = float(df["rating"].mean()) if total else 0.0
density = density_per_km2(total, radius_m)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Results", total)
k2.metric("Avg Score", f"{avg_score:.1f}/100")
k3.metric("Avg Rating", f"{avg_rating:.2f}⭐")
k4.metric("Density", f"{density:.1f} / km²")

st.divider()
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
        mime="text/csv"
    )

with right:
    st.subheader("Map view")
    map_df = df[["lat", "lng"]].rename(columns={"lat": "latitude", "lng": "longitude"})
    st.map(map_df, zoom=12)

st.divider()
st.caption("Demo note: This app uses generated sample data for portfolio presentation.")
