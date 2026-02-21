# app.py — Location Intelligence Dashboard (Actor #7 compatible)
# Paste this whole file to replace your current app.py

from __future__ import annotations

import math
import os
from io import BytesIO
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


# =============================================================================
# Streamlit config + styling
# =============================================================================
st.set_page_config(page_title="Location Intelligence Dashboard", layout="wide")

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


# =============================================================================
# Constants / mappings
# =============================================================================
UI_TO_ACTOR_CATEGORY_IDS = {
    "pharmacy": ["health_pharmacy"],
    "restaurant": ["food_restaurant"],
    "fast_food": ["food_fast_food"],
    "cafe": ["food_cafe"],
    "bar_pub": ["food_bar_pub"],
    "supermarket": ["retail_supermarket"],
    "convenience": ["retail_convenience"],
    "grocery": ["retail_grocery"],
    "bakery": ["retail_bakery"],
    "clinic": ["health_clinic"],
    "hospital": ["health_hospital"],
    "dentist": ["health_dentist"],
}

def _wanted_category_ids(ui_category: str) -> List[str]:
    ui = (ui_category or "").strip().lower()
    if not ui:
        return []
    return UI_TO_ACTOR_CATEGORY_IDS.get(ui, [ui])


# =============================================================================
# Helper functions
# =============================================================================
def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "export") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_secret(key: str, default: str = "") -> str:
    v = os.getenv(key)
    if v:
        return v
    try:
        if hasattr(st, "secrets"):
            return str(st.secrets.get(key, default))
    except Exception:
        return default
    return default


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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


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


def compute_pressure_and_risk(density: float, comp_share: float, avg_score: float) -> Tuple[float, float]:
    dens_n = clamp01(density / 20.0)      # demo scaling
    comp_n = clamp01(comp_share)          # 0..1
    quality_n = clamp01(avg_score / 100.0)
    pressure = 100.0 * (0.55 * comp_n + 0.45 * dens_n)
    risk = 100.0 * (0.55 * (1.0 - quality_n) + 0.45 * (0.55 * comp_n + 0.45 * dens_n))
    return float(round(pressure, 0)), float(round(risk, 0))
def _pct_norm(series: pd.Series, p_low=25, p_high=90, default=0.55) -> float:
    """Normalize a numeric series to 0..1 using percentiles (robust to outliers)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return default
    lo = np.percentile(s, p_low)
    hi = np.percentile(s, p_high)
    if hi <= lo:
        return default
    return float(np.clip((s - lo) / (hi - lo), 0, 1).mean())


def compute_attractiveness_index(
    df_in: pd.DataFrame,
    *,
    radius_m: int,
    density_per_km2_val: float,
    comp_share_val: float,
    target_density: float = 25.0,
) -> pd.Series:
    """
    Returns a per-row Attractiveness Index (0..100).
    Uses: distance_m, rating, review_count, is_competitor (if present).
    """
    df = df_in.copy()

    # --- Demand (D): from review_count if available ---
    if "review_count" in df.columns:
        reviews = np.log1p(pd.to_numeric(df["review_count"], errors="coerce"))
        D = _pct_norm(reviews, p_low=25, p_high=90, default=0.55)
    else:
        D = 0.55

    # --- Accessibility (A): from distance_m ---
    if "distance_m" in df.columns:
        dist = pd.to_numeric(df["distance_m"], errors="coerce")
        A_row = 1.0 - (dist / float(radius_m))
        A_row = A_row.clip(0, 1).fillna(0.6)
    else:
        A_row = pd.Series(0.6, index=df.index)

    # --- Quality (Q): from rating ---
    if "rating" in df.columns:
        rating = pd.to_numeric(df["rating"], errors="coerce")
        Q_row = ((rating - 3.0) / 2.0).clip(0, 1).fillna(0.6)
    else:
        Q_row = pd.Series(0.6, index=df.index)

    # --- Competition (C): from density + competitor share ---
    density_norm = float(np.clip(density_per_km2_val / float(target_density), 0, 1))
    C = float(np.clip(0.6 * density_norm + 0.4 * float(comp_share_val), 0, 1))

    # Weights
    wD, wA, wQ = 0.45, 0.25, 0.30
    wC = 0.70

    base = (wD * D) + (wA * A_row) + (wQ * Q_row)
    score01 = (base * (1.0 - wC * C)).clip(0, 1)

    return (100.0 * score01).round(1)

def apply_competitor_keywords(df: pd.DataFrame, keywords_csv: str) -> pd.DataFrame:
    keywords = [k.strip().lower() for k in (keywords_csv or "").split(",") if k.strip()]
    if not keywords or df is None or df.empty:
        return df

    name_col = _pick_first(df, ["name", "title", "place_name", "poi_name", "business_name", "entity_name"])
    if not name_col:
        return df

    name_l = df[name_col].astype(str).str.lower()
    hit = pd.Series(False, index=df.index)
    for k in keywords:
        hit = hit | name_l.str.contains(k, na=False)

    out = df.copy()
    if "is_competitor" not in out.columns:
        out["is_competitor"] = False
    out["is_competitor"] = out["is_competitor"].astype(bool) | hit
    return out


def opportunity_recommendation(
    opp_index: float,
    density: float,
    comp_share: float,
    pressure_0_100: float,
    risk_0_100: float,
    avg_score: float,
):
    pct = opp_index * 100.0
    comp_pct = comp_share * 100.0

    if density >= 60:
        saturation_label = "highly saturated"
    elif density >= 15:
        saturation_label = "moderately saturated"
    elif density >= 6:
        saturation_label = "partially underserved"
    else:
        saturation_label = "structurally underserved"

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


# =============================================================================
# Apify loaders
# =============================================================================
def load_apify_dataset_items(dataset_id: str, apify_token: Optional[str], limit: int = 5000) -> pd.DataFrame:
    dataset_id = (dataset_id or "").strip()
    if not dataset_id:
        return pd.DataFrame()

    params = {"clean": "true", "format": "json", "limit": str(limit)}
    if apify_token:
        params["token"] = apify_token.strip()

    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"

    try:
        r = requests.get(url, params=params, timeout=45)
        if r.status_code != 200:
            st.error(f"Apify dataset fetch failed ({r.status_code}): {r.text[:500]}")
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list):
            st.error(f"Unexpected dataset response type: {type(data)}")
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Apify dataset fetch exception: {e}")
        return pd.DataFrame()


def load_apify_run(run_id: str, apify_token: str) -> dict:
    run_id = (run_id or "").strip()
    if not run_id:
        return {}
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    params = {"token": apify_token}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            st.error(f"Apify run fetch failed ({r.status_code}): {r.text[:500]}")
            return {}
        js = r.json()
        if isinstance(js, dict) and "data" in js and isinstance(js["data"], dict):
            return js["data"]
        return js if isinstance(js, dict) else {}
    except Exception as e:
        st.error(f"Apify run fetch exception: {e}")
        return {}


# =============================================================================
# Schema normalization + mode detection
# =============================================================================
def normalize_actor7_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    # POI schema normalization
    if "poi_lat" in out.columns and "lat" not in out.columns:
        out["lat"] = out["poi_lat"]
    if "poi_lon" in out.columns and "lng" not in out.columns:
        out["lng"] = out["poi_lon"]
    if "poi_name" in out.columns and "name" not in out.columns:
        out["name"] = out["poi_name"]

    # generic alternatives
    if "latitude" in out.columns and "lat" not in out.columns:
        out["lat"] = out["latitude"]
    if "longitude" in out.columns and "lng" not in out.columns:
        out["lng"] = out["longitude"]
    if "lon" in out.columns and "lng" not in out.columns:
        out["lng"] = out["lon"]

    # enforce numeric if present
    for c in ["lat", "lng", "anchor_lat", "anchor_lon"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # coverage numeric
    if "radius_m" in out.columns:
        out["radius_m"] = pd.to_numeric(out["radius_m"], errors="coerce")
    if "services_count" in out.columns:
        out["services_count"] = pd.to_numeric(out["services_count"], errors="coerce").fillna(0)

    # ensure category_id exists (canonical key)
    if "category_id" in out.columns:
        out["category_id"] = out["category_id"].astype(str)
    elif "category" in out.columns:
        out["category_id"] = out["category"].astype(str)
    elif "category_name" in out.columns:
        out["category_id"] = out["category_name"].astype(str)

    # coverage_adequate fallback
    if "coverage_adequate" not in out.columns and "coverage_status" in out.columns:
        out["coverage_adequate"] = out["coverage_status"].astype(str).str.lower().eq("adequate")

    return out


def detect_dataset_mode(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "unknown"
    cols = set(df.columns)

    # Coverage rows
    if {"services_count", "coverage_status"}.issubset(cols) or "coverage_adequate" in cols:
        return "coverage"

    # POI rows
    if {"lat", "lng"}.issubset(cols) or {"poi_lat", "poi_lon"}.issubset(cols):
        return "poi"

    return "unknown"


def compute_center_from_df(df: pd.DataFrame, fallback_lat: float, fallback_lon: float) -> Tuple[float, float]:
    if df is None or df.empty:
        return fallback_lat, fallback_lon

    if {"lat", "lng"}.issubset(df.columns):
        lat_s = pd.to_numeric(df["lat"], errors="coerce").dropna()
        lon_s = pd.to_numeric(df["lng"], errors="coerce").dropna()
        if len(lat_s) and len(lon_s):
            return float(lat_s.mean()), float(lon_s.mean())

    if {"poi_lat", "poi_lon"}.issubset(df.columns):
        lat_s = pd.to_numeric(df["poi_lat"], errors="coerce").dropna()
        lon_s = pd.to_numeric(df["poi_lon"], errors="coerce").dropna()
        if len(lat_s) and len(lon_s):
            return float(lat_s.mean()), float(lon_s.mean())

    if {"anchor_lat", "anchor_lon"}.issubset(df.columns):
        lat_s = pd.to_numeric(df["anchor_lat"], errors="coerce").dropna()
        lon_s = pd.to_numeric(df["anchor_lon"], errors="coerce").dropna()
        if len(lat_s) and len(lon_s):
            return float(lat_s.mean()), float(lon_s.mean())

    return fallback_lat, fallback_lon


def ensure_distance_m(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if "distance_m" in out.columns:
        out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
        return out

    # choose center: if anchor coords exist, use first anchor row as center
    c_lat, c_lon = center_lat, center_lon
    if {"anchor_lat", "anchor_lon"}.issubset(out.columns):
        a_lat_s = pd.to_numeric(out["anchor_lat"], errors="coerce").dropna()
        a_lon_s = pd.to_numeric(out["anchor_lon"], errors="coerce").dropna()
        if len(a_lat_s) and len(a_lon_s):
            c_lat, c_lon = float(a_lat_s.iloc[0]), float(a_lon_s.iloc[0])

    if {"lat", "lng"}.issubset(out.columns):
        lat_s = pd.to_numeric(out["lat"], errors="coerce")
        lon_s = pd.to_numeric(out["lng"], errors="coerce")

        def _dist_row(x):
            try:
                if pd.isna(x["lat"]) or pd.isna(x["lng"]):
                    return np.nan
                return float(haversine_km(c_lat, c_lon, float(x["lat"]), float(x["lng"])) * 1000.0)
            except Exception:
                return np.nan

        tmp = pd.DataFrame({"lat": lat_s, "lng": lon_s})
        out["distance_m"] = tmp.apply(_dist_row, axis=1)
        return out

    out["distance_m"] = np.nan
    return out


# =============================================================================
# Category / coverage filtering + unified analysis_df
# =============================================================================
def filter_coverage_df(df: pd.DataFrame, *, ui_category: str, radius_m: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()

    # normalize numeric radius + services_count
    if "radius_m" in work.columns:
        work["radius_m"] = pd.to_numeric(work["radius_m"], errors="coerce")
    if "services_count" in work.columns:
        work["services_count"] = pd.to_numeric(work["services_count"], errors="coerce").fillna(0)

    # filter radius
    if "radius_m" in work.columns and radius_m is not None:
        work = work[work["radius_m"] == float(radius_m)].copy()

    wanted = _wanted_category_ids(ui_category)
    if wanted and "category_id" in work.columns:
        work = work[work["category_id"].astype(str).isin(wanted)].copy()

    # coverage_adequate fallback
    if "coverage_adequate" not in work.columns and "coverage_status" in work.columns:
        work["coverage_adequate"] = work["coverage_status"].astype(str).str.lower().eq("adequate")

    return work.reset_index(drop=True)


def category_filter_best_effort(df: pd.DataFrame, ui_category: str) -> pd.DataFrame:
    """Best-effort POI filtering by category based on available columns."""
    if df is None or df.empty:
        return df

    wanted = _wanted_category_ids(ui_category)
    work = df.copy()

    # Strongest: category_id exact match (actor7 may store it per POI)
    if "category_id" in work.columns and wanted:
        m = work["category_id"].astype(str).isin(wanted)
        if m.any():
            return work[m].copy()

    # Next: category / type columns containing tokens
    probe_cols = [c for c in ["category", "categories", "types", "primaryType", "place_types", "category_name"] if c in work.columns]
    if probe_cols and wanted:
        wanted_l = [w.lower() for w in wanted] + [(ui_category or "").strip().lower()]
        mask = pd.Series(False, index=work.index)
        for c in probe_cols:
            s = work[c].astype(str).str.lower()
            for w in wanted_l:
                if w:
                    mask = mask | s.str.contains(w, na=False)
        if mask.any():
            return work[mask].copy()

    return work


def build_analysis_df(
    df: pd.DataFrame,
    *,
    mode: str,
    category: str,
    radius_m: int,
    show_competitors: bool,
    center_lat: float,
    center_lon: float,
) -> pd.DataFrame:
    """
    Returns ONE dataframe that the whole app uses for:
    - total
    - density
    - competitor share
    - table
    - export
    So KPIs and tables never disagree.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    if mode == "coverage":
        work = normalize_actor7_schema(work)
        work = filter_coverage_df(work, ui_category=category, radius_m=radius_m)
        if "services_count" in work.columns:
            work["services_count"] = pd.to_numeric(work["services_count"], errors="coerce").fillna(0)
        return work.reset_index(drop=True)

    # POI mode
    work = normalize_actor7_schema(work)
    work = ensure_distance_m(work, center_lat, center_lon)
    work = category_filter_best_effort(work, category)

    # radius filter
    if "distance_m" in work.columns:
        dist = pd.to_numeric(work["distance_m"], errors="coerce")
        if dist.notna().any():
            work = work[dist <= int(radius_m)].copy()

    # competitor filter toggle
    if not show_competitors and "is_competitor" in work.columns:
        work = work[work["is_competitor"] == False].copy()

    return work.reset_index(drop=True)


# =============================================================================
# Demo data generator (synthetic)
# =============================================================================
def make_demo_points(center_lat, center_lon, category, radius_m, n, seed=42):
    rng = np.random.default_rng(int(seed))
    r_deg = (radius_m / 1000.0) / 111.0
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
            "lat": np.round(lats, 6),
            "lng": np.round(lons, 6),
            "rating": np.round(ratings, 1),
            "review_count": reviews,
            "is_competitor": competitor_flag,
        }
    )

    df["distance_m"] = df.apply(
        lambda r: float(haversine_km(center_lat, center_lon, r["lat"], r["lng"]) * 1000.0),
        axis=1,
    )

    dist_norm = (df["distance_m"] / float(radius_m)).clip(0, 1)
    rating_norm = ((df["rating"] - 3.0) / 2.0).clip(0, 1)
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

    return df.sort_values(["score", "review_count"], ascending=False).reset_index(drop=True)


# =============================================================================
# Snapshots
# =============================================================================
def build_multi_radius_snapshot_poi(df: pd.DataFrame, radius_list=(300, 500, 1000, 1500, 2000)) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "distance_m" in work.columns:
        work["distance_m"] = pd.to_numeric(work["distance_m"], errors="coerce")

    rows = []
    for r in radius_list:
        sub = work
        if "distance_m" in sub.columns and sub["distance_m"].notna().any():
            sub = sub[sub["distance_m"] <= int(r)]

        tot = int(len(sub))
        avg_score_r = safe_mean(sub, "score")
        avg_rating_r = safe_mean(sub, "rating")
        density_r = density_per_km2(tot, int(r))
        comp_share_r = float(sub["is_competitor"].mean()) if tot and "is_competitor" in sub.columns else 0.0
        opp_r = clamp01((avg_score_r / 100.0) * (1.0 - comp_share_r))
        pressure_r, risk_r = compute_pressure_and_risk(density_r, comp_share_r, avg_score_r)

        rows.append(
            {
                "Radius (m)": int(r),
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


def build_multi_radius_snapshot_coverage(df: pd.DataFrame, *, category: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            [{"Radius (m)": r, "Results": 0, "Density (/km²)": 0.0, "Opportunity %": 100.0, "Pressure": 0.0, "Risk": 55.0}
             for r in [300, 500, 1000, 1500, 2000]]
        )

    work = normalize_actor7_schema(df).copy()

    if "radius_m" in work.columns:
        work["radius_m"] = pd.to_numeric(work["radius_m"], errors="coerce")
    if "services_count" in work.columns:
        work["services_count"] = pd.to_numeric(work["services_count"], errors="coerce").fillna(0)

    if "coverage_adequate" not in work.columns and "coverage_status" in work.columns:
        work["coverage_adequate"] = work["coverage_status"].astype(str).str.lower().eq("adequate")

    wanted = _wanted_category_ids(category)
    if wanted and "category_id" in work.columns:
        work = work[work["category_id"].astype(str).isin(wanted)]

    radii = sorted({int(x) for x in work["radius_m"].dropna().tolist()}) if "radius_m" in work.columns else []
    if not radii:
        radii = [300, 500, 1000, 1500, 2000]

    rows = []
    for r in radii:
        sub = work[work["radius_m"] == float(r)] if "radius_m" in work.columns else work

        results = int(pd.to_numeric(sub.get("services_count", 0), errors="coerce").fillna(0).sum()) if len(sub) else 0

        # Density fix fallback: if results==0 but nearest_distance_m says at least one inside radius
        if results == 0 and len(sub) and "nearest_distance_m" in sub.columns:
            nd = pd.to_numeric(sub["nearest_distance_m"], errors="coerce").dropna()
            if len(nd) and float(nd.min()) <= float(r):
                results = 1

        dens = density_per_km2(int(results), int(r))
        opp = 1.0 if results == 0 else float(1.0 / (1.0 + dens))
        opp_pct = 100.0 * opp

        # Simple pressure/risk proxies (coverage mode lacks POI quality fields)
        pressure = float(100.0 * (dens / (dens + 50.0))) if results > 0 else 0.0
        risk = 55.0 if results == 0 else max(15.0, 55.0 - min(40.0, results * 2.0))

        rows.append(
            {
                "Radius (m)": int(r),
                "Results": int(results),
                "Density (/km²)": round(dens, 2),
                "Opportunity %": round(opp_pct, 0),
                "Pressure": round(pressure, 0),
                "Risk": round(risk, 0),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PDF builder (Executive Memo)
# =============================================================================
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
    is_coverage: bool,
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
    muted = ParagraphStyle("Muted", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#666666"))

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
            str(int(total)),
            f"{float(avg_score):.1f}/100",
            "—" if is_coverage else f"{float(avg_rating):.2f}",
            f"{float(density):.1f}",
            f"{float(opp_index) * 100:.0f}%",
            f"{float(comp_share) * 100:.0f}%",
            f"{float(pressure_0_100):.0f}/100",
            f"{float(risk_0_100):.0f}/100",
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
        cols = list(snap.columns)
        snap_data = [cols] + snap.astype(str).values.tolist()
        col_w = [max(18 * mm, int(180 * mm / max(1, len(cols))))] * len(cols)
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
        story.append(Paragraph("Priority rows (Top 10)" if is_coverage else "Top results (Top 10)", h_style))

        if is_coverage:
            cols = [c for c in ["anchor_name", "radius_m", "category_name", "services_count", "nearest_distance_m", "coverage_status"] if c in top_df.columns]
            if not cols:
                cols = list(top_df.columns)
            tdf = top_df[cols].copy().head(10)
        else:
            cols = [c for c in ["name", "rating", "review_count", "distance_m", "score", "is_competitor"] if c in top_df.columns]
            if not cols:
                cols = list(top_df.columns)
            tdf = top_df[cols].copy().head(10)
            if "name" in tdf.columns:
                tdf["name"] = tdf["name"].astype(str).str.slice(0, 40)

        top_data = [cols] + tdf.astype(str).values.tolist()
        top_col_w = [max(18 * mm, int(180 * mm / max(1, len(cols))))] * len(cols)
        top_table = Table(top_data, colWidths=top_col_w)
        top_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                    ("TOPPADDING", (0, 0), (-1, 0), 5),
                ]
            )
        )
        story.append(top_table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: Signals are illustrative; validate with your own competitive definitions and local knowledge.", muted))

    doc.build(story)
    return buf.getvalue()


# =============================================================================
# Sidebar controls
# =============================================================================
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

category = st.sidebar.selectbox("Category", list(UI_TO_ACTOR_CATEGORY_IDS.keys()), index=0)
radius_m = st.sidebar.selectbox("Radius (meters)", [300, 500, 1000, 1500, 2000], index=2)
n_points = st.sidebar.slider("Base # results @ 1000m (demo)", 10, 120, 45, step=5)

st.sidebar.divider()
st.sidebar.subheader("Competitor Definition (optional)")
competitor_keywords_csv = st.sidebar.text_input("Competitor keywords (comma-separated)", "")
show_competitors = st.sidebar.checkbox("Include competitors", value=True)

st.sidebar.divider()
st.sidebar.subheader("Data source")
data_mode = st.sidebar.radio("Choose data source", ["Demo (synthetic)", "Apify Run ID", "Apify Dataset ID"], index=0)

apify_token = get_secret("APIFY_TOKEN", "")
run_id = ""
dataset_id = ""

if data_mode == "Apify Run ID":
    run_id = st.sidebar.text_input("Apify Run ID", "")
elif data_mode == "Apify Dataset ID":
    dataset_id = st.sidebar.text_input("Apify Dataset ID", "")

st.sidebar.divider()
debug_raw = st.sidebar.checkbox("Debug: show raw dataset head", value=False)
debug_density = st.sidebar.checkbox("Debug: density inputs", value=False)

st.sidebar.divider()
st.sidebar.subheader("Refresh")
if "demo_nonce" not in st.session_state:
    st.session_state["demo_nonce"] = 0
if st.sidebar.button("🔄 Regenerate demo data"):
    st.session_state["demo_nonce"] += 1
    st.cache_data.clear()

centers = {
    "Los Angeles (Downtown)": (34.052235, -118.243683),
    "Washington, DC": (38.907192, -77.036873),
    "New York (Midtown)": (40.754932, -73.984016),
    "Berlin (Mitte)": (52.520008, 13.404954),
}
fallback_center_lat, fallback_center_lon = centers[preset]
seed = abs(hash((preset, city, category, radius_m, n_points, st.session_state["demo_nonce"]))) % (10**6)


# =============================================================================
# Data load
# =============================================================================
if data_mode == "Demo (synthetic)":
    # Build ONE master demo dataset at max radius (so snapshot + KPIs are consistent)
    max_r = 2000

    # Keep density roughly comparable: n_points means "around 1000m"
    # Area scales ~ r^2, so scale total points accordingly
    n_master = int(round(n_points * (max_r / 1000.0) ** 2))
    n_master = max(20, min(5000, n_master))

    df = make_demo_points(
        fallback_center_lat,
        fallback_center_lon,
        category,
        max_r,
        n_master,
        seed=seed,
    )
    df = apply_competitor_keywords(df, competitor_keywords_csv)
else:
    if not apify_token:
        st.error("Missing APIFY_TOKEN. Add it in Streamlit secrets (or env var APIFY_TOKEN).")
        st.stop()

    if data_mode == "Apify Run ID":
        if not run_id.strip():
            st.info("Enter an Apify Run ID to load results.")
            st.stop()
        run = load_apify_run(run_id.strip(), apify_token)
        dataset_id = (run.get("defaultDatasetId") or "").strip()

    if not (dataset_id or "").strip():
        st.error("Provide a valid Apify Dataset ID (or a Run ID that resolves to one).")
        st.stop()

    df = load_apify_dataset_items(dataset_id.strip(), apify_token, limit=5000)
    df = normalize_actor7_schema(df)
    df = apply_competitor_keywords(df, competitor_keywords_csv)

mode = detect_dataset_mode(df)

data_center_lat, data_center_lon = compute_center_from_df(df, fallback_center_lat, fallback_center_lon)

if debug_raw:
    st.write("Columns:", [] if df is None else df.columns.tolist())
    st.dataframe(df.head(30) if df is not None else pd.DataFrame(), use_container_width=True)


# =============================================================================
# KPI compute (coverage vs poi) — STABLE (density + competitor% + snapshot/export aligned)
# =============================================================================
is_coverage = mode == "coverage"

analysis_df = build_analysis_df(
    df,
    mode=mode,
    category=category,
    radius_m=radius_m,
    show_competitors=show_competitors,
    center_lat=data_center_lat,
    center_lon=data_center_lon,
)

# Debug
st.sidebar.write("DEBUG mode:", mode)
st.sidebar.write("DEBUG analysis rows:", 0 if analysis_df is None else len(analysis_df))
if analysis_df is not None and "distance_m" in analysis_df.columns:
    dist_num = pd.to_numeric(analysis_df["distance_m"], errors="coerce")
    st.sidebar.write("DEBUG distance_m NaN %:", float(dist_num.isna().mean()))

# Defaults
total = 0
density = 0.0
comp_share = 0.0
avg_score = 0.0
avg_rating = 0.0
opp_index = 1.0 if is_coverage else 0.0
pressure_0_100, risk_0_100 = 0.0, 55.0

if analysis_df is not None and not analysis_df.empty:
    if is_coverage:
        sc = pd.to_numeric(analysis_df.get("services_count", 0), errors="coerce").fillna(0)
        total = int(sc.sum())

        # Density fallback: services_count==0 but nearest_distance_m suggests at least 1 within radius
        if total == 0 and "nearest_distance_m" in analysis_df.columns:
            nd = pd.to_numeric(analysis_df["nearest_distance_m"], errors="coerce").dropna()
            if len(nd) and float(nd.min()) <= float(radius_m):
                total = 1

        density = density_per_km2(total, int(radius_m))

        if "competitor_share" in analysis_df.columns:
            cs = pd.to_numeric(analysis_df["competitor_share"], errors="coerce").fillna(0)
            comp_share = float(cs.max())
            comp_share = comp_share / 100.0 if comp_share > 1.0 else comp_share
        else:
            comp_share = 0.0

        adequate_any = False
        if "coverage_adequate" in analysis_df.columns:
            adequate_any = bool(analysis_df["coverage_adequate"].astype(bool).any())

        avg_score = 100.0 if adequate_any else 0.0
        avg_rating = 0.0

        opp_index = 1.0 if total == 0 else float(1.0 / (1.0 + density))
        pressure_0_100, risk_0_100 = compute_pressure_and_risk(density, comp_share, avg_score)

        if debug_density:
            cols = [c for c in ["radius_m", "category_id", "category_name", "services_count", "nearest_distance_m", "competitor_share"] if c in analysis_df.columns]
            if cols:
                st.sidebar.dataframe(analysis_df[cols].head(12), use_container_width=True)
            st.sidebar.write("DEBUG total (sum services_count):", total)
            st.sidebar.write("DEBUG density:", density)
    else:
        total = int(len(analysis_df))
        density = density_per_km2(total, int(radius_m))

        comp_share = float(analysis_df["is_competitor"].mean()) if total and "is_competitor" in analysis_df.columns else 0.0

        score_sum = pd.to_numeric(analysis_df.get("score", 0), errors="coerce").fillna(0).sum() if "score" in analysis_df.columns else 0
        if "score" not in analysis_df.columns or score_sum == 0:
            density_score = 90 - min(80, density * 2.0)
            comp_penalty = comp_share * 40.0
            fallback_score = max(10.0, min(90.0, density_score - comp_penalty))
            analysis_df["score"] = float(fallback_score)

        rating_sum = pd.to_numeric(analysis_df.get("rating", 0), errors="coerce").fillna(0).sum() if "rating" in analysis_df.columns else 0
        if "rating" not in analysis_df.columns or rating_sum == 0:
            analysis_df["rating"] = 3.5

        avg_score = safe_mean(analysis_df, "score")
        avg_rating = safe_mean(analysis_df, "rating")

        opp_index = clamp01((avg_score / 100.0) * (1.0 - comp_share))
        pressure_0_100, risk_0_100 = compute_pressure_and_risk(density, comp_share, avg_score)


# =============================================================================
# Header
# =============================================================================
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
    Data source: <b>{data_mode}</b> • Mode: <b>{mode}</b> • Category: <b>{category}</b> • Radius: <b>{radius_m}m</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# =============================================================================
# Tabs
# =============================================================================
tab_overview, tab_results, tab_method = st.tabs(["📌 Overview", "📋 Results", "🧠 Method"])

# Pre-compute recommendation (used in Overview + PDF)
rec = opportunity_recommendation(
    opp_index=opp_index,
    density=density,
    comp_share=comp_share,
    pressure_0_100=pressure_0_100,
    risk_0_100=risk_0_100,
    avg_score=avg_score,
)

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Results", int(total))
    c2.metric("Avg Score", f"{avg_score:.1f}/100", score_label(avg_score))
    c3.metric("Avg Rating", "—" if is_coverage else f"{avg_rating:.2f}⭐")
    c4.metric("Density (/km²)", f"{density:.1f}")
    c5.metric("Opportunity", f"{opp_index * 100:.0f}%", pct_label(opp_index))

    st.write("")
    k1, k2, k3 = st.columns(3)
    k1.metric("Competitive Pressure", f"{pressure_0_100:.0f}/100", pressure_label(pressure_0_100))
    k2.metric("Entry Risk", f"{risk_0_100:.0f}/100", pressure_label(risk_0_100))
    k3.metric("Competitor Share", f"{comp_share * 100:.0f}%")

    st.write("")
    st.markdown("### Executive Insight")

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

    st.write("")
    st.markdown("### Snapshot")



    if data_mode == "Demo (synthetic)":
        # IMPORTANT: use the SAME master df (generated once at load time)
        # so Snapshot matches the Overview KPIs for the selected radius.
        snap = build_multi_radius_snapshot_poi(df, radius_list=(300, 500, 1000, 1500, 2000))

    elif is_coverage:
        snap = build_multi_radius_snapshot_coverage(df, category=category)  # keep full df for multi-radius

    else:
        snap = build_multi_radius_snapshot_poi(df)  # keep full df for multi-radius


    st.dataframe(snap, use_container_width=True, height=240)

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
- Use exports in the **Results** tab for reporting-ready output
"""
        )

        if data_mode != "Demo (synthetic)":
            st.info("Apify mode: Dataset/Run decides the real location. Area preset changes labels only.")

    with right:
        st.markdown("### Map preview")

        # Prefer plotting analysis_df if it has coords; else fallback to raw df
        map_source = analysis_df if (analysis_df is not None and not analysis_df.empty) else df

        has_poi_coords = (
            map_source is not None
            and not map_source.empty
            and {"lat", "lng"}.issubset(map_source.columns)
            and pd.to_numeric(map_source["lat"], errors="coerce").notna().any()
            and pd.to_numeric(map_source["lng"], errors="coerce").notna().any()
        )

        has_anchor_coords = (
            df is not None
            and not df.empty
            and {"anchor_lat", "anchor_lon"}.issubset(df.columns)
            and pd.to_numeric(df["anchor_lat"], errors="coerce").notna().any()
            and pd.to_numeric(df["anchor_lon"], errors="coerce").notna().any()
        )

        if has_poi_coords:
            map_df = map_source[["lat", "lng"]].copy()
            map_df["latitude"] = pd.to_numeric(map_df["lat"], errors="coerce")
            map_df["longitude"] = pd.to_numeric(map_df["lng"], errors="coerce")
            map_df = map_df[["latitude", "longitude"]].dropna()
            if len(map_df) > 2000:
                map_df = map_df.sample(n=2000, random_state=42)
            st.map(map_df, zoom=12)

        elif has_anchor_coords:
            a_lat = float(pd.to_numeric(df["anchor_lat"], errors="coerce").dropna().iloc[0])
            a_lon = float(pd.to_numeric(df["anchor_lon"], errors="coerce").dropna().iloc[0])
            st.map(pd.DataFrame({"latitude": [a_lat], "longitude": [a_lon]}), zoom=12)
            st.info("Showing anchor location (coverage rows don't include POI coordinates).")

        else:
            st.map(pd.DataFrame({"latitude": [data_center_lat], "longitude": [data_center_lon]}), zoom=12)
            st.info("No mappable columns found; showing dataset-derived center point.")

        st.markdown(
            f"<div class='muted small'>Dataset center: {data_center_lat:.5f}, {data_center_lon:.5f} • Radius: {radius_m}m</div>",
            unsafe_allow_html=True,
        )


with tab_results:
    st.markdown("### Results table")

    if analysis_df is None or analysis_df.empty:
        st.info("No results for the selected filters (mode/category/radius).")
    else:
        preferred_cols = [
            "name", "poi_name",
            "category_name", "category_id",
            "lat", "lng", "poi_lat", "poi_lon",
            "distance_m",
            "is_competitor",
            "score", "rating",
            "review_count",
            "address", "website", "phone",
            "services_count", "nearest_distance_m", "competitor_share",
            "coverage_status", "coverage_adequate",
            "anchor_name", "anchor_lat", "anchor_lon",
        ]
        show_cols = [c for c in preferred_cols if c in analysis_df.columns]
        if not show_cols:
            show_cols = list(analysis_df.columns)[:18]

        results_view = analysis_df[show_cols].copy()
        st.dataframe(results_view, use_container_width=True, height=520)

        st.markdown("### Export (what you see is what you export)")
        export_left, export_right = st.columns([1, 1])

        with export_left:
            csv_bytes = results_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV (shown results)",
                data=csv_bytes,
                file_name=f"results_{mode}_{category}_{int(radius_m)}m.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_right:
            xlsx_bytes = df_to_xlsx_bytes(results_view, sheet_name="results")
            st.download_button(
                "⬇️ Download XLSX (shown results)",
                data=xlsx_bytes,
                file_name=f"results_{mode}_{category}_{int(radius_m)}m.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.write("")
    st.markdown("### Executive Memo (PDF)")
    if df is None or df.empty:
        st.info("No results to export.")
    else:
        # choose top rows based on mode
        if is_coverage:
            work = normalize_actor7_schema(df).copy()
            if "coverage_adequate" not in work.columns and "coverage_status" in work.columns:
                work["coverage_adequate"] = work["coverage_status"].astype(str).str.lower().eq("adequate")
            if "services_count" in work.columns:
                work["services_count"] = pd.to_numeric(work["services_count"], errors="coerce").fillna(0)

            sort_cols = [c for c in ["coverage_adequate", "services_count", "nearest_distance_m"] if c in work.columns]
            if sort_cols:
                # prefer inadequate + low service counts (opportunity signals)
                top = work.sort_values(sort_cols, ascending=[True, True, False][: len(sort_cols)]).head(12)
            else:
                top = work.head(12)
            snap_pdf = build_multi_radius_snapshot_coverage(df, category=category)
        else:
            sort_cols = [c for c in ["score", "rating", "review_count"] if c in (analysis_df.columns if analysis_df is not None else [])]
            if analysis_df is not None and not analysis_df.empty and sort_cols:
                top = analysis_df.sort_values(sort_cols, ascending=False).head(10)
            elif analysis_df is not None and not analysis_df.empty:
                top = analysis_df.head(10)
            else:
                top = df.head(10)
            snap_pdf = build_multi_radius_snapshot_poi(df)

        pdf_bytes = build_executive_memo_pdf(
            city=city,
            preset=preset,
            category=category,
            radius_m=radius_m,
            total=int(total),
            avg_score=float(avg_score),
            avg_rating=float(avg_rating),
            density=float(density),
            opp_index=float(opp_index),
            comp_share=float(comp_share),
            pressure_0_100=float(pressure_0_100),
            risk_0_100=float(risk_0_100),
            rec=rec,
            snapshot_df=snap_pdf,
            top_df=top,
            is_coverage=bool(is_coverage),
        )

        st.download_button(
            "⬇️ Download Executive Memo (PDF)",
            data=pdf_bytes,
            file_name=f"executive_memo_{preset.replace(' ', '_')}_{category}_{radius_m}m.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


with tab_method:
    st.markdown(
        """
### Method (what the dashboard computes)

**POI mode (rows with coordinates):**
- Supports Actor #7 POI schema: `poi_lat/poi_lon/poi_name` (auto-normalized to `lat/lng/name`)
- Filters by **Radius** using `distance_m` (computed if missing)
- Density = POIs per km² within selected radius
- Competitor share = % rows flagged as competitor (via keywords)

**Coverage mode (catchment metrics):**
- Filters by **category_id** using UI mapping (e.g., `pharmacy` → `health_pharmacy`)
- Results = `SUM(services_count)` across anchors for the selected radius & category
- Density = results / km²
- Map uses `anchor_lat/anchor_lon` because coverage rows do not include POI coordinates

**Apify Run ID vs Dataset ID**
- **Run ID** identifies a specific Actor run in Apify (from which we auto-resolve its `defaultDatasetId`).
- **Dataset ID** directly identifies the dataset to load.
"""
    )