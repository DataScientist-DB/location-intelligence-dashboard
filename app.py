import math
import os
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Optional: Apify integration (safe if installed; otherwise ignored)
try:
    from apify_client import ApifyClient  # pip install apify-client
except Exception:
    ApifyClient = None

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
.hr {
  height: 1px;
  background: rgba(49,51,63,0.10);
  margin: 10px 0 12px 0;
}
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

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(df[col].mean())

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

def pressure_label(val: float) -> str:
    if val < 35:
        return "🟢 Low"
    elif val < 70:
        return "🟠 Medium"
    return "🔴 High"

def compute_pressure_and_risk(density: float, comp_share: float, avg_score: float) -> tuple[float, float]:
    """
    Competitive Pressure (0..100): competition + density-driven intensity.
    Entry Risk (0..100): pressure + weak quality signals.
    """
    dens_n = clamp01(density / 20.0)  # demo scaling
    comp_n = clamp01(comp_share)      # already 0..1
    quality_n = clamp01(avg_score / 100.0)

    pressure = 100.0 * (0.55 * comp_n + 0.45 * dens_n)
    risk = 100.0 * (0.55 * (1.0 - quality_n) + 0.45 * (0.55 * comp_n + 0.45 * dens_n))
    return float(round(pressure, 0)), float(round(risk, 0))

def opportunity_recommendation(
    opp_index: float,
    density: float,
    comp_share: float,
    pressure_0_100: float,
    risk_0_100: float,
    avg_score: float,
):
    """
    Executive-style recommendation based on:
    - Opportunity index (0..1)
    - Market density (per km2)
    - Competitor share (0..1)
    - Competitive pressure (0..100)
    - Entry risk (0..100)
    """
    pct = opp_index * 100.0
    comp_pct = comp_share * 100.0

    # Density bands tuned for demo (avoid calling 3.6/km² "highly saturated")
    if density >= 60:
        saturation_label = "highly saturated"
    elif density >= 15:
        saturation_label = "moderately saturated"
    elif density >= 6:
        saturation_label = "partially underserved"
    else:
        saturation_label = "structurally underserved"

    # Market badge (short)
    if pct < 30:
        market_badge = "🔴 High-risk market"
    elif pct < 60:
        market_badge = "🟠 Selective market"
    else:
        market_badge = "🟢 Attractive market"

    if pct < 30:
        headline = "Limited Entry Attractiveness"
        text = (
            f"{market_badge}. The area appears {saturation_label} with elevated competitive pressure "
            f"({comp_pct:.0f}% competitor share). Market entry risk is high and returns depend on clear differentiation."
        )
        action = "Re-check adjacent micro-zones, test a niche format, or improve differentiation before committing."
        color = "#ffe5e5"
    elif pct < 60:
        headline = "Selective Opportunity"
        text = (
            f"{market_badge}. The location shows {saturation_label} conditions with moderate competitive presence "
            f"({comp_pct:.0f}% competitor share). Performance will depend on micro-location quality and positioning."
        )
        action = "Shortlist high-footfall corners, test proximity to anchors, and benchmark rents prior to decision."
        color = "#fff4e0"
    else:
        headline = "Favorable Market Entry Conditions"
        text = (
            f"{market_badge}. The area appears {saturation_label} with manageable competitive intensity "
            f"({comp_pct:.0f}% competitor share). Market signals support expansion or a new location feasibility."
        )
        action = "Proceed with due diligence, access checks, and rental benchmarking; validate demand with a small pilot."
        color = "#e6f4ea"

    basis = (
        f"Opportunity {pct:.0f}% • Avg score {avg_score:.1f} • Competitor share {comp_pct:.0f}% • "
        f"Density {density:.1f}/km² • Pressure {pressure_0_100:.0f}/100 • Risk {risk_0_100:.0f}/100"
    )

    return {"headline": headline, "text": text, "action": action, "basis": basis, "color": color}

def apply_competitor_keywords(df: pd.DataFrame, keywords_csv: str) -> pd.DataFrame:
    """
    Mark rows as competitors if 'name' contains any keyword.
    (In demo data names are generic; in production this becomes powerful.)
    """
    keywords = [k.strip().lower() for k in (keywords_csv or "").split(",") if k.strip()]
    if not keywords or "name" not in df.columns:
        return df
    name_l = df["name"].astype(str).str.lower()
    hit = pd.Series(False, index=df.index)
    for k in keywords:
        hit = hit | name_l.str.contains(k, na=False)
    df = df.copy()
    if "is_competitor" not in df.columns:
        df["is_competitor"] = False
    df["is_competitor"] = df["is_competitor"] | hit
    return df

# -----------------------------
# DEMO DATA GENERATOR (NO CACHING!)
# -----------------------------
def make_demo_points(center_lat, center_lon, category, radius_m, n, seed=42):
    """
    Generate synthetic POIs. IMPORTANT: no @st.cache_data decorator here.
    """
    rng = np.random.default_rng(int(seed))

    # Spread points in a circle around the center
    r_deg = (radius_m / 1000.0) / 111.0  # approx degrees per km
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = r_deg * np.sqrt(rng.uniform(0, 1, n))

    lats = center_lat + radii * np.cos(angles)
    lons = center_lon + radii * np.sin(angles) / np.cos(np.deg2rad(center_lat))

    names = [f"{category.title()} #{i+1}" for i in range(n)]
    ratings = np.clip(rng.normal(4.2, 0.35, n), 3.0, 5.0)
    reviews = np.clip(rng.normal(180, 90, n).astype(int), 5, 1200)

    # base competitor probability (demo)
    competitor_flag = rng.choice([True, False], size=n, p=[0.35, 0.65])

    df = pd.DataFrame(
        {
            "name": names,
            "category": category,
            "lat": np.round(lats, 6),
            "lng": np.round(lons, 6),
            "rating": np.round(ratings, 1),
            "review_count": reviews,
            "is_competitor": competitor_flag,
        }
    )

    df["distance_m"] = df.apply(
        lambda r: int(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000),
        axis=1,
    )

    # Composite score (0..100)
    dist_norm = (df["distance_m"] / float(radius_m)).clip(0, 1)  # 0 close → 1 far
    rating_norm = ((df["rating"] - 3.0) / 2.0).clip(0, 1)        # 3..5 → 0..1
    review_norm = (np.log1p(df["review_count"]) / np.log1p(1200)).clip(0, 1)

    rng2 = np.random.default_rng(int(seed) + 999)
    noise = rng2.normal(0, 4.0, len(df))

    score = (
        15
        + 45 * rating_norm
        + 18 * review_norm
        + 22 * (1 - dist_norm)
        - 7 * df["is_competitor"].astype(int)
        + noise
    )
    df["score"] = np.clip(np.round(score), 0, 100).astype(int)

    # Coverage proxy (0..1)
    df["coverage_pct"] = np.clip((df["score"] / 100) * rng.uniform(0.7, 1.1, n), 0, 1).round(2)

    return df.sort_values(["score", "review_count"], ascending=False).reset_index(drop=True)

# -----------------------------
# PDF (Executive Memo)
# -----------------------------
def build_executive_memo_pdf(
    *,
    city: str,
    preset: str,
    category: str,
    radius_m: int,
    total: int,
    avg_score: float,
    avg_rating: float,
    density: float,
    opp_index: float,
    comp_share: float,
    pressure_0_100: float,
    risk_0_100: float,
    rec: dict,
    snapshot_df: pd.DataFrame,
    top_df: pd.DataFrame,
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Location Intelligence Memo",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=16, leading=20, spaceAfter=10)
    h_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=10, leading=14)
    muted = ParagraphStyle(
        "Muted", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#666666")
    )

    story = []
    story.append(Paragraph("Location Intelligence — Executive Memo", title_style))
    story.append(
        Paragraph(
            f"<b>Study:</b> {preset} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Category:</b> {category} "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; <b>Radius:</b> {radius_m}m",
            body,
        )
    )
    story.append(Paragraph(f"<b>City / Area label:</b> {city}", muted))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Key Metrics", h_style))
    kpi_data = [
        ["Results", "Avg Score", "Avg Rating", "Density (/km²)", "Opportunity", "Competitor Share", "Pressure", "Risk"],
        [
            str(total),
            f"{avg_score:.1f}/100",
            f"{avg_rating:.2f}",
            f"{density:.1f}",
            f"{opp_index * 100:.0f}%",
            f"{comp_share * 100:.0f}%",
            f"{pressure_0_100:.0f}/100",
            f"{risk_0_100:.0f}/100",
        ],
    ]
    t = Table(kpi_data, colWidths=[22*mm, 24*mm, 22*mm, 26*mm, 22*mm, 30*mm, 22*mm, 18*mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Executive Insight", h_style))
    story.append(Paragraph(f"<b>{rec.get('headline','')}</b>", body))
    story.append(Spacer(1, 4))
    story.append(Paragraph(rec.get("text", ""), body))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Recommended next step:</b> {rec.get('action','')}", body))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Basis:</b> {rec.get('basis','')}", muted))

    if snapshot_df is not None and not snapshot_df.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Multi-radius Snapshot", h_style))
        snap = snapshot_df.copy()
        snap_cols = list(snap.columns)
        snap_data = [snap_cols] + snap.astype(str).values.tolist()
        col_w = [18*mm, 20*mm, 18*mm, 20*mm, 20*mm, 24*mm, 22*mm, 22*mm, 18*mm, 16*mm]
        col_w = col_w[: len(snap_cols)]
        tt = Table(snap_data, colWidths=col_w)
        tt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F9FAFB")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                    ("TOPPADDING", (0, 0), (-1, 0), 5),
                ]
            )
        )
        story.append(tt)

    if top_df is not None and not top_df.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Top Results (Top 10)", h_style))
        cols = [c for c in ["name", "rating", "review_count", "distance_m", "score", "is_competitor"] if c in top_df.columns]
        tdf = top_df[cols].copy()
        tdf["name"] = tdf["name"].astype(str).str.slice(0, 38)
        tdf = tdf.head(10)

        top_data = [cols] + tdf.astype(str).values.tolist()
        top_col_w = []
        for c in cols:
            if c == "name":
                top_col_w.append(62 * mm)
            elif c in ("review_count", "distance_m"):
                top_col_w.append(24 * mm)
            else:
                top_col_w.append(20 * mm)

        top_table = Table(top_data, colWidths=top_col_w)
        top_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                    ("TOPPADDING", (0, 0), (-1, 0), 5),
                ]
            )
        )
        story.append(top_table)

    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "Note: Demo-mode signals are illustrative. In production, connect to live POI sources and your competitor definitions.",
            muted,
        )
    )

    doc.build(story)
    return buf.getvalue()

# -----------------------------
# Multi-radius snapshot (NO caching!)
# -----------------------------
def build_multi_radius_snapshot(
    center_lat: float,
    center_lon: float,
    category: str,
    n_points: int,
    base_seed: int,
    competitor_keywords_csv: str,
    include_competitors: bool,
    demo_density_mode: str,
    radii=(300, 500, 1000, 1500, 2000),
) -> pd.DataFrame:
    """
    Snapshot modes:
    1) Constant N: same generated N for each radius → density changes with radius (obvious)
    2) Constant density: generated N scales with area → density ~constant (baseline comparison)
    3) Market gradient: density varies by radius (more realistic “CBD vs outskirts” vibe)

    Fixes:
    - 300/500 no longer freeze at the hard min (10)
    - 1500/2000 no longer both hit the same hard max (350)
    - "Generated N" varies naturally, and "Results" can differ slightly (drop-rate)
    """
    rows = []
    base_r = 1000  # reference radius for n_points

    for r in radii:
        # Strong per-radius seed
        dynamic_seed = abs(hash((center_lat, center_lon, category, r, n_points, base_seed))) % (10**9)
        rng = np.random.default_rng(dynamic_seed)

        # -----------------------------
        # 1) Compute baseline expected N
        # -----------------------------
        if demo_density_mode.startswith("Constant N"):
            base_n = float(n_points)

        elif demo_density_mode.startswith("Constant density"):
            base_n = float(n_points) * (r / base_r) ** 2

        else:
            # Market gradient: higher density close-in, lower further out
            gradient = (base_r / float(r)) ** 0.7  # 300m bigger, 2000m smaller
            base_n = float(n_points) * (r / base_r) ** 2 * gradient

        # -----------------------------
        # 2) Add noise WITHOUT “small-radius freeze” or “big-radius ceiling ties”
        #    (Poisson repeats too often for small lambdas; use normal here)
        # -----------------------------
        sigma = max(2.0, base_n * 0.18)
        noisy_n = int(round(rng.normal(loc=base_n, scale=sigma)))

        # -----------------------------
        # 3) Radius-aware bounds (instead of hard 10..350)
        # -----------------------------
        min_n = max(6, int(round(n_points * 0.10)))  # not always 10
        # max grows with radius & base_n (prevents 1500/2000 both capping)
        max_n = int(round(max(base_n * 2.2, n_points * (r / base_r) ** 2 * 3.0)))
        max_n = max(max_n, min_n + 12)

        scaled_n = int(np.clip(noisy_n, min_n, max_n))

        # -----------------------------
        # 4) Generate data
        # -----------------------------
        dfr = make_demo_points(center_lat, center_lon, category, r, scaled_n, seed=dynamic_seed)
        dfr = apply_competitor_keywords(dfr, competitor_keywords_csv)

        if not include_competitors and "is_competitor" in dfr.columns:
            dfr = dfr[dfr["is_competitor"] == False].reset_index(drop=True)

        # OPTIONAL realism: let Results differ from Generated N a bit
        # (simulate dedupe/invalids/filter fallout)
        drop_rate = float(rng.uniform(0.03, 0.12))
        keep_frac = max(0.75, 1.0 - drop_rate)  # never drop too much
        dfr = dfr.sample(frac=keep_frac, random_state=int(dynamic_seed % (2**32 - 1))).reset_index(drop=True)

        # -----------------------------
        # 5) Metrics
        # -----------------------------
        tot = len(dfr)
        avg_score_r = safe_mean(dfr, "score")
        avg_rating_r = safe_mean(dfr, "rating")
        density_r = density_per_km2(tot, r)
        comp_share_r = float(dfr["is_competitor"].mean()) if tot and "is_competitor" in dfr.columns else 0.0

        opp_r = clamp01((avg_score_r / 100.0) * (1.0 - comp_share_r))
        pressure_r, risk_r = compute_pressure_and_risk(density_r, comp_share_r, avg_score_r)

        rows.append(
            {
                "Radius (m)": r,
                "Generated N": int(scaled_n),
                "Results": int(tot),
                "Avg Score": round(avg_score_r, 1),
                "Avg Rating": round(avg_rating_r, 2),
                "Density (/km²)": round(density_r, 1),
                "Competitor %": int(round(comp_share_r * 100, 0)),
                "Opportunity %": int(round(opp_r * 100, 0)),
                "Pressure": int(round(pressure_r, 0)),
                "Risk": int(round(risk_r, 0)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Optional: Apify fetch helpers (cached is OK here)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def apify_fetch_run(run_id: str, token: str):
    if ApifyClient is None:
        raise RuntimeError("apify-client not installed. Add apify-client to requirements.txt")
    client = ApifyClient(token)
    return client.run(run_id).get()

@st.cache_data(show_spinner=False, ttl=60)
def apify_fetch_kv_json(store_id: str, key: str, token: str):
    if ApifyClient is None:
        raise RuntimeError("apify-client not installed. Add apify-client to requirements.txt")
    client = ApifyClient(token)
    rec = client.key_value_store(store_id).get_record(key)
    return (rec or {}).get("value")

@st.cache_data(show_spinner=False, ttl=60)
@st.cache_data(show_spinner=False, ttl=60)
def apify_fetch_dataset_items(dataset_id: str, token: str, limit: int = 5000) -> pd.DataFrame:
    if ApifyClient is None:
        raise RuntimeError("apify-client not installed. Add apify-client to requirements.txt")

    client = ApifyClient(token)
    items: list = []
    offset = 0
    page_limit = 1000

    while True:
        resp = client.dataset(dataset_id).list_items(limit=page_limit, offset=offset)

        # --- Normalize response across apify-client versions ---
        if resp is None:
            batch = []

        # Newer/older versions may return dict-like payload
        elif isinstance(resp, dict):
            batch = resp.get("items") or resp.get("data") or []

        # Some versions return an object with `.items`
        elif hasattr(resp, "items") and not isinstance(resp, list):
            batch = getattr(resp, "items") or []

        # Some versions may return list directly
        else:
            batch = resp

        # Safety: ensure batch is a list
        if batch is None:
            batch = []
        elif isinstance(batch, dict):
            batch = [batch]
        else:
            batch = list(batch)

        items.extend(batch)

        # Stop conditions
        if len(batch) < page_limit or len(items) >= limit:
            break

        offset += page_limit

    return pd.DataFrame(items[:limit])

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Analysis Parameters")

preset = st.sidebar.selectbox(
    "Area",
    ["Los Angeles (Downtown)", "Washington, DC", "New York (Midtown)", "Berlin (Mitte)"],
    key="area_preset",
)

city_map = {
    "Los Angeles (Downtown)": "Los Angeles, CA",
    "Washington, DC": "Washington, DC",
    "New York (Midtown)": "New York, NY",
    "Berlin (Mitte)": "Berlin, Germany",
}
city = city_map.get(preset, preset)
st.sidebar.caption(f"City / Area label: {city}")


# Auto-sync city label with preset
city_map = {
    "Los Angeles (Downtown)": "Los Angeles, CA",
    "Washington, DC": "Washington, DC",
    "New York (Midtown)": "New York, NY",
    "Berlin (Mitte)": "Berlin, Germany",
}

city = city_map.get(preset, preset)




category = st.sidebar.selectbox("Category", ["pharmacy", "restaurant", "hospital", "school", "grocery"])
radius_m = st.sidebar.selectbox("Radius (meters)", [300, 500, 1000, 1500, 2000], index=2)
n_points = st.sidebar.slider("Base # results @ 1000m", 10, 120, 45, step=5)

st.sidebar.divider()
st.sidebar.subheader("Competitor Definition (optional)")
competitor_keywords_csv = st.sidebar.text_input("Competitor keywords (comma-separated)", "")

show_competitors = st.sidebar.checkbox("Include competitors", value=True)

st.sidebar.divider()
st.sidebar.subheader("Export")
export_name = st.sidebar.text_input("CSV file name", "location_intelligence_export.csv")

# Optional: data source (Demo vs Apify)
st.sidebar.divider()
st.sidebar.subheader("Data source")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Demo (synthetic)", "Apify Run ID", "Apify Dataset ID"],
    index=0,
)
apify_token = (st.secrets.get("APIFY_TOKEN", "") if hasattr(st, "secrets") else "") or os.getenv("APIFY_TOKEN", "")
run_id = ""
dataset_id = ""
if data_mode == "Apify Run ID":
    run_id = st.sidebar.text_input("Apify Run ID", "")
elif data_mode == "Apify Dataset ID":
    dataset_id = st.sidebar.text_input("Apify Dataset ID", "")

st.sidebar.divider()
st.sidebar.subheader("Refresh")
if "demo_nonce" not in st.session_state:
    st.session_state["demo_nonce"] = 0
if st.sidebar.button("🔄 Regenerate demo data"):
    st.session_state["demo_nonce"] += 1
    # This clears apify caches too; that's ok for demo/troubleshooting.
    st.cache_data.clear()

centers = {
    "Los Angeles (Downtown)": (34.052235, -118.243683),
    "Washington, DC": (38.907192, -77.036873),
    "New York (Midtown)": (40.754932, -73.984016),
    "Berlin (Mitte)": (52.520008, 13.404954),
}
center_lat, center_lon = centers[preset]

st.sidebar.subheader("Demo realism")
demo_density_mode = st.sidebar.radio(
    "Demo mode behavior",
    ["Constant N (density changes with radius)", "Constant density (N scales with area)", "Market gradient (density varies by radius)"],
    index=1,
)

# Seed must change with parameters + nonce (prevents "frozen" snapshot)
seed = abs(hash((preset, city, category, radius_m, n_points, st.session_state["demo_nonce"]))) % (10**6)

# -----------------------------
# Data
# -----------------------------
summary = None

if data_mode == "Demo (synthetic)":
    df = make_demo_points(center_lat, center_lon, category, radius_m, n_points, seed=seed)
    df = apply_competitor_keywords(df, competitor_keywords_csv)
else:
    if not apify_token:
        st.error("Missing APIFY_TOKEN. Add it in Streamlit secrets.")
        st.stop()

    if data_mode == "Apify Run ID":
        if not run_id.strip():
            st.info("Enter an Apify Run ID to load results.")
            st.stop()
        run = apify_fetch_run(run_id.strip(), apify_token)
        dataset_id = (run.get("defaultDatasetId") or "").strip()
        store_id = (run.get("defaultKeyValueStoreId") or "").strip()
        if store_id:
            summary = apify_fetch_kv_json(store_id, "RUN_SUMMARY.json", apify_token)

    if not (dataset_id or "").strip():
        st.error("Could not resolve dataset ID. Provide a valid Dataset ID or Run ID.")
        st.stop()

    df = apify_fetch_dataset_items(dataset_id.strip(), apify_token, limit=5000)

    # normalize expected columns
    rename_map = {}
    if "latitude" in df.columns and "lat" not in df.columns:
        rename_map["latitude"] = "lat"
    if "longitude" in df.columns and "lng" not in df.columns:
        rename_map["longitude"] = "lng"
    if rename_map:
        df = df.rename(columns=rename_map)

    df = apply_competitor_keywords(df, competitor_keywords_csv)

# Ensure expected columns exist (robust)
defaults = {
    "name": "",
    "category": category,
    "lat": np.nan,
    "lng": np.nan,
    "rating": 0.0,
    "review_count": 0,
    "distance_m": 0,
    "score": 0,
    "is_competitor": False,
}
for col, default in defaults.items():
    if col not in df.columns:
        df[col] = default

# Apply competitor toggle
if not show_competitors and "is_competitor" in df.columns:
    df = df[df["is_competitor"] == False].reset_index(drop=True)

total = len(df)
avg_score = safe_mean(df, "score")
avg_rating = safe_mean(df, "rating")
density = density_per_km2(total, radius_m)
comp_share = float(df["is_competitor"].mean()) if total and "is_competitor" in df.columns else 0.0

opp_index = clamp01((avg_score / 100.0) * (1.0 - comp_share))
pressure_0_100, risk_0_100 = compute_pressure_and_risk(density, comp_share, avg_score)

# -----------------------------
# Header (Professional positioning)
# -----------------------------
st.markdown(
    f"""
<div class="card">
  <div style="margin-bottom:8px;">
    <span class="badge">Decision Support</span>
    <span class="badge">Site Selection</span>
    <span class="badge">Market Saturation</span>
    <span class="badge">Competitor Pressure</span>
  </div>

  <h1 style="margin:0;">🧭 Location Intelligence Dashboard</h1>

  <div class="muted" style="margin-top:6px;">
    Executive-ready micro-market screening for <b>site selection</b>, <b>retail strategy</b>, and <b>feasibility briefs</b>.
    Anchor: <b>{preset}</b> • Category: <b>{category}</b> • Radius: <b>{radius_m}m</b>
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
    # KPI row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Results", total)
    c2.metric("Avg Score", f"{avg_score:.1f}/100", score_label(avg_score))
    c3.metric("Avg Rating", f"{avg_rating:.2f}⭐")
    c4.metric("Density (/km²)", f"{density:.1f}")
    c5.metric("Opportunity", f"{opp_index * 100:.0f}%", pct_label(opp_index))

    # KPI row 2
    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.metric("Competitive Pressure", f"{pressure_0_100:.0f}/100", pressure_label(pressure_0_100))
    k2.metric("Entry Risk", f"{risk_0_100:.0f}/100", pressure_label(risk_0_100))
    k3.metric("Competitor Share", f"{comp_share * 100:.0f}%")

    # Executive Insight
    st.write("")
    st.markdown("### Executive Insight")
    rec = opportunity_recommendation(
        opp_index=opp_index,
        density=density,
        comp_share=comp_share,
        pressure_0_100=pressure_0_100,
        risk_0_100=risk_0_100,
        avg_score=avg_score,
    )

    st.markdown(
        f"""
<div style="
    border-radius:16px;
    padding:18px;
    background:{rec['color']};
    border:1px solid rgba(0,0,0,0.05);
">
  <div style="font-weight:850; font-size:1.15rem; margin-bottom:8px;">
    {rec['headline']}
  </div>

  <div style="line-height:1.65; font-size:1.02rem; margin-bottom:10px;">
    {rec['text']}
  </div>

  <div class="muted small" style="line-height:1.6; margin-bottom:8px;">
    <b>Recommended next step:</b> {rec['action']}
  </div>

  <div class="muted small" style="line-height:1.6;">
    <b>Basis:</b> {rec['basis']}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Multi-radius snapshot (always recompute; no caching)
    st.write("")
    st.markdown("### Multi-radius snapshot")
    snap = build_multi_radius_snapshot(
        center_lat=center_lat,
        center_lon=center_lon,
        category=category,
        n_points=n_points,
        base_seed=seed,
        competitor_keywords_csv=competitor_keywords_csv,
        include_competitors=show_competitors,
        demo_density_mode=demo_density_mode,
    )

    st.dataframe(snap, use_container_width=True, height=240)

    # Business value + map
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

        if data_mode != "Demo (synthetic)" and summary:
            st.write("")
            st.markdown("### Run summary (from Actor)")
            st.json(summary)

    with right:
        st.markdown("### Map preview")
        if total and {"lat", "lng"}.issubset(df.columns):
            map_df = df[["lat", "lng"]].rename(columns={"lat": "latitude", "lng": "longitude"}).dropna()
            st.map(map_df, zoom=12)
        else:
            st.info("No rows available for the current filters.")
        st.markdown(
            f"<div class='muted small'>Anchor: {preset} • Radius: {radius_m}m • City label: {city}</div>",
            unsafe_allow_html=True,
        )

    # Score distribution
    st.write("")
    st.markdown("### Score distribution")
    if total and "score" in df.columns:
        hist = df[["score"]].copy()
        hist["bucket"] = (hist["score"] // 10) * 10
        dist = hist.groupby("bucket").size().reset_index(name="count").sort_values("bucket").set_index("bucket")
        st.bar_chart(dist)
    else:
        st.info("Score distribution is unavailable (no results).")

with tab_results:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Results table")
        view_cols = ["name", "category", "rating", "review_count", "distance_m", "score", "is_competitor"]
        existing = [c for c in view_cols if c in df.columns]
        st.dataframe(df[existing], use_container_width=True, height=520)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=export_name,
            mime="text/csv",
        )

    with right:
        st.subheader("Top performers")
        if total:
            top = df.sort_values(["score", "rating", "review_count"], ascending=False).head(10)

            # Build PDF memo
            pdf_bytes = build_executive_memo_pdf(
                city=city,
                preset=preset,
                category=category,
                radius_m=radius_m,
                total=total,
                avg_score=avg_score,
                avg_rating=avg_rating,
                density=density,
                opp_index=opp_index,
                comp_share=comp_share,
                pressure_0_100=pressure_0_100,
                risk_0_100=risk_0_100,
                rec=rec,
                snapshot_df=snap,
                top_df=top,
            )

            st.download_button(
                "⬇️ Download Executive Memo (PDF)",
                data=pdf_bytes,
                file_name=f"executive_memo_{preset.replace(' ', '_')}_{category}_{radius_m}m.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            top_cols = [c for c in ["name", "rating", "review_count", "distance_m", "score", "is_competitor"] if c in top.columns]
            st.dataframe(top[top_cols], use_container_width=True, height=350)
        else:
            st.info("No results to display.")

        st.subheader("Competitor share")
        if total and "is_competitor" in df.columns:
            comp = pd.DataFrame(
                {
                    "segment": ["Competitors", "Non-competitors"],
                    "count": [int(df["is_competitor"].sum()), int((~df["is_competitor"]).sum())],
                }
            ).set_index("segment")
            st.bar_chart(comp)
        else:
            st.info("Competitor share is unavailable (no results).")

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

    st.markdown("### KPI logic (high-level)")
    st.markdown(
        """
- **Avg Score (0–100):** composite of rating, review volume, distance-to-anchor, competitor penalty  
- **Density (/km²):** count divided by circle area (πr²)  
- **Competitor share:** % of rows labeled as competitor  
- **Opportunity:** (AvgScore/100) × (1 − competitorShare)  
- **Competitive Pressure:** combination of density and competitor share  
- **Entry Risk:** pressure + low quality signals (low avg score)
"""
    )

    st.markdown("### Notes on demo realism")
    st.markdown(
        """
- In demo mode, the snapshot scales the **generated count** with radius area, so density behaves intuitively across radii.  
- In real (Apify) mode, density reflects the actual dataset size returned by your Actor for the chosen radius/filtering.
"""
    )

st.caption("Note: Data shown is sample-formatted for demonstration. The dashboard can be connected to a live backend when needed.")
