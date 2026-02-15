import math
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

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

@st.cache_data(show_spinner=False)
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

    # base competitor probability (demo)
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

    # Composite score (0..100)
    dist_norm = (df["distance_m"] / radius_m).clip(0, 1)                 # 0 close → 1 far
    rating_norm = ((df["rating"] - 3.0) / 2.0).clip(0, 1)               # 3..5 → 0..1
    review_norm = (np.log1p(df["review_count"]) / np.log1p(1200)).clip(0, 1)

    rng2 = np.random.default_rng(seed + 999)
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

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def compute_pressure_and_risk(density: float, comp_share: float, avg_score: float) -> tuple[float, float]:
    """
    Competitive Pressure (0..100): competition + density-driven intensity.
    Entry Risk (0..100): pressure + weak quality signals.
    """
    # Density scaling: 0..~20 is typical; clamp stronger markets
    dens_n = clamp01(density / 20.0)
    comp_n = clamp01(comp_share)  # already 0..1
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
        """
        Returns PDF bytes for an executive-ready memo.
        Uses reportlab (Streamlit Cloud-safe).
        """
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
        title_style = ParagraphStyle(
            "Title2",
            parent=styles["Title"],
            fontSize=16,
            leading=20,
            spaceAfter=10,
        )
        h_style = ParagraphStyle(
            "H",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        )
        body = ParagraphStyle(
            "Body2",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
        )
        muted = ParagraphStyle(
            "Muted",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#666666"),
        )

        story = []

        # Title + context
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

        # KPI table
        story.append(Paragraph("Key Metrics", h_style))
        kpi_data = [
            ["Results", "Avg Score", "Avg Rating", "Density (/km²)", "Opportunity", "Competitor Share", "Pressure",
             "Risk"],
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
        t = Table(kpi_data, colWidths=[22 * mm, 24 * mm, 22 * mm, 26 * mm, 22 * mm, 30 * mm, 22 * mm, 18 * mm])
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

        # Executive insight box
        story.append(Paragraph("Executive Insight", h_style))
        story.append(Paragraph(f"<b>{rec.get('headline', '')}</b>", body))
        story.append(Spacer(1, 4))
        story.append(Paragraph(rec.get("text", ""), body))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Recommended next step:</b> {rec.get('action', '')}", body))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Basis:</b> {rec.get('basis', '')}", muted))

        # Snapshot table
        if snapshot_df is not None and not snapshot_df.empty:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Multi-radius Snapshot", h_style))

            snap = snapshot_df.copy()
            snap_cols = list(snap.columns)
            snap_data = [snap_cols] + snap.astype(str).values.tolist()

            # Reasonable widths for A4
            col_w = [18 * mm, 18 * mm, 20 * mm, 20 * mm, 24 * mm, 22 * mm, 22 * mm, 18 * mm, 16 * mm]
            col_w = col_w[: len(snap_cols)]  # safe if you change columns later
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

        # Top results
        if top_df is not None and not top_df.empty:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Top Results (Top 10)", h_style))
            cols = [c for c in ["name", "rating", "review_count", "distance_m", "score", "is_competitor"] if
                    c in top_df.columns]
            tdf = top_df[cols].copy()

            # Make it compact + safe for PDF
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

        # Footer note
        story.append(Spacer(1, 12))
        story.append(
            Paragraph(
                "Note: Demo-mode signals are illustrative. In production, connect to live POI sources and your competitor definitions.",
                muted,
            )
        )

        doc.build(story)
        return buf.getvalue()

    # Density bands tuned for your demo (avoid calling 3.6/km² "highly saturated")
    if density >= 60:
        saturation_label = "highly saturated"
    elif density >= 15:
        saturation_label = "moderately saturated"
    elif density >= 6:
        saturation_label = "partially underserved"
    else:
        saturation_label = "structurally underserved"

    # Narrative templates
    if pct < 30:
        return {
            "headline": "Limited Entry Attractiveness",
            "text": (
                f"The area appears {saturation_label} with elevated competitive pressure "
                f"({comp_pct:.0f}% competitor share). Market entry risk is high and returns depend on clear differentiation."
            ),
            "action": "Re-check adjacent micro-zones, test a niche format, or improve offer differentiation before committing.",
            "basis": f"Opportunity {pct:.0f}% • Avg score {avg_score:.1f} • Competitor share {comp_pct:.0f}% • Density {density:.1f}/km² • Pressure {pressure_0_100:.0f}/100 • Risk {risk_0_100:.0f}/100",
            "color": "#ffe5e5",  # light red
        }

    if pct < 60:
        return {
            "headline": "Selective Opportunity",
            "text": (
                f"The area shows {saturation_label} conditions with moderate competitive presence "
                f"({comp_pct:.0f}% competitor share). Performance will depend on micro-location quality and brand positioning."
            ),
            "action": "Shortlist high-footfall corners, test proximity to anchors, and conduct rental benchmarking prior to decision.",
            "basis": f"Opportunity {pct:.0f}% • Avg score {avg_score:.1f} • Competitor share {comp_pct:.0f}% • Density {density:.1f}/km² • Pressure {pressure_0_100:.0f}/100 • Risk {risk_0_100:.0f}/100",
            "color": "#fff4e0",  # light orange
        }

    return {
        "headline": "Favorable Market Entry Conditions",
        "text": (
            f"The area appears {saturation_label} with manageable competitive intensity "
            f"({comp_pct:.0f}% competitor share). Market signals support expansion or new location feasibility."
        ),
        "action": "Proceed with site due diligence, access checks, and rental benchmarking; validate demand with a small pilot.",
        "basis": f"Opportunity {pct:.0f}% • Avg score {avg_score:.1f} • Competitor share {comp_pct:.0f}% • Density {density:.1f}/km² • Pressure {pressure_0_100:.0f}/100 • Risk {risk_0_100:.0f}/100",
        "color": "#e6f4ea",  # light green
    }

def apply_competitor_keywords(df: pd.DataFrame, keywords_csv: str) -> pd.DataFrame:
    """
    Optional: If competitor keywords are given, mark rows as competitors if name contains keyword.
    (Demo dataset names are generic; in production this becomes powerful.)
    """
    keywords = [k.strip().lower() for k in (keywords_csv or "").split(",") if k.strip()]
    if not keywords:
        return df
    name_l = df["name"].astype(str).str.lower()
    hit = pd.Series(False, index=df.index)
    for k in keywords:
        hit = hit | name_l.str.contains(k, na=False)
    df = df.copy()
    df["is_competitor"] = df["is_competitor"] | hit
    return df

def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(df[col].mean())

def build_multi_radius_snapshot(
    center_lat: float,
    center_lon: float,
    category: str,
    n_points: int,
    seed: int,
    competitor_keywords_csv: str,
    include_competitors: bool,
    radii=(300, 500, 1000, 1500, 2000),
) -> pd.DataFrame:
    rows = []
    for r in radii:
        dfr = make_demo_points(center_lat, center_lon, category, r, n_points, seed=seed)
        dfr = apply_competitor_keywords(dfr, competitor_keywords_csv)

        if not include_competitors:
            dfr = dfr[dfr["is_competitor"] == False].reset_index(drop=True)

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
                "Results": tot,
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

st.sidebar.divider()
st.sidebar.subheader("Competitor Definition (optional)")
competitor_keywords_csv = st.sidebar.text_input("Competitor keywords (comma-separated)", "")

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
df = apply_competitor_keywords(df, competitor_keywords_csv)

if not show_competitors:
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

    # KPI row 2 (new)
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

    # Multi-radius snapshot
    st.write("")
    st.markdown("### Multi-radius snapshot")
    snap = build_multi_radius_snapshot(
        center_lat=center_lat,
        center_lon=center_lon,
        category=category,
        n_points=n_points,
        seed=seed,
        competitor_keywords_csv=competitor_keywords_csv,
        include_competitors=show_competitors,
    )
    st.dataframe(snap, use_container_width=True, height=220)

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
        if total:
            map_df = df[["lat", "lng"]].rename(columns={"lat": "latitude", "lng": "longitude"})
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
        dist = (
            hist.groupby("bucket")
            .size()
            .reset_index(name="count")
            .sort_values("bucket")
            .set_index("bucket")
        )
        st.bar_chart(dist)
    else:
        st.info("Score distribution is unavailable (no results).")

with tab_results:
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Results table")
        view_cols = ["name", "category", "rating", "review_count", "distance_m", "score", "is_competitor"]
        # robust selection
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

            st.sidebar.download_button(
                "⬇️ Download Executive Memo (PDF)",
                data=pdf_bytes,
                file_name=f"executive_memo_{preset.replace(' ', '_')}_{category}_{radius_m}m.pdf",
                mime="application/pdf",
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

st.caption("Note: Data shown is sample-formatted for demonstration. The dashboard can be connected to a live backend when needed.")
