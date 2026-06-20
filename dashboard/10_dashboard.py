"""
Metropolitan Police demand dashboard (CBL Group 16 proof of concept).

Two geographies behind one toggle (sidebar → Area):
  * London: the team's final 4-feature index (severity-weighted crime, VIIRS
    night-lights, employment deprivation, resolution rate), 4,994 LSOAs,
    rendered as an LSOA-level choropleth with live weather, the event overlay
    and police-station coverage gaps.
  * UK (England): the national 3-feature *structural* index (night-lights,
    resolution rate, employment deprivation, deliberately no crime volume),
    33,755 LSOAs, rendered as a Local-Authority-District choropleth (LSOA
    polygons don't scale to 33k in the browser). The London-only panels
    switch off automatically.

The PRIORITY TIERS are the Phase 6 supervised tiers (boundaries calibrated
on a 24-month fit against the next 12 months' realised top-15% hotspots,
AUC 0.92 London / 0.83 England). They are the only thing called a tier, used
by the filters, KPIs and allocation. Staged by fetch_team_model.py from the
CBL-16 final pipelines.

Run from the repo root:
    pip install -r dashboard/requirements.txt
    cd dashboard
    python fetch_dashboard_assets.py   # one-off: LSOA boundaries + stations
    streamlit run 10_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
ASSETS = ROOT / "dashboard_assets"

LSOA_GEOJSON = ASSETS / "london_lsoa.geojson"
LAD_GEOJSON = ASSETS / "england_lad.geojson"
# Same boundaries with the 33 Greater London LADs dissolved into one polygon
# (built by build_london_merged_geojson.py). The national map shows London as a
# single region, coloured distinctly from the rest of England for comparison.
LAD_GEOJSON_MERGED = ASSETS / "england_lad_londonmerged.geojson"
GREATER_LONDON = "Greater London"
# Distinct fill/stroke for the Greater London region so it stands apart from the
# red "rest of England" ramp (blue is CVD-safe against the Reds palette).
LONDON_REGION_FILL = "#2B6CB0"
LONDON_REGION_LINE = "#11365E"
POLICE_GEOJSON = ASSETS / "london_police_stations.geojson"
STYLE_CSS = ASSETS / "style.css"

# Dynamic event overlay lives in 12_event_overlay.py (same dir). Imported via
# importlib because the module name starts with a digit. Only its runtime path
# (load_overlay_tables / compute_overlay_config) is used here: pure pandas, no
# geopandas, so it adds nothing to the Space's memory footprint. London only.
import importlib
import sys as _sys
_sys.path.insert(0, str(ROOT))
try:
    _overlay = importlib.import_module("12_event_overlay")
except Exception:
    _overlay = None


@st.cache_data(show_spinner="Loading event overlay…")
def get_overlay_tables():
    """The four prebuilt overlay tables, or None if they're not shipped."""
    if _overlay is None:
        return None
    try:
        return _overlay.load_overlay_tables()
    except Exception:
        return None


def _resolve(name: str, extra_dirs: list[Path]) -> Path:
    """First existing path for `name` across the candidate dirs, else the
    first candidate (so the caller can raise a clear FileNotFoundError)."""
    candidates = [d / name for d in extra_dirs]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# ----- geography profiles -----
# Everything that differs between the London model and the England model is
# captured here, so the body of the app can stay geography-agnostic and just
# read `GEO[...]`.

# Tier role per the 10 Jun review:
#   `tier` = Phase 6 SUPERVISED tiers (1 highest, 4 low), the only
#            operational priority layer. Boundaries calibrated against the
#            LSOAs that became top-15% demand in the following 12 months
#            (AUC 0.92 London / 0.83 England). Size-honest: Tier 1
#            CONTAINS 25.3% of London LSOAs (34.8% England); it CAPTURED
#            81.7% (77.2%) of the realised future hotspots in validation.
LONDON_TIER_LABEL = {1: "Highest", 2: "High", 3: "Moderate", 4: "Low"}
LONDON_TIER_COLOR = {
    # Okabe-Ito colourblind-safe palette. Distinguishable for deuteranopia,
    # protanopia and tritanopia, and in greyscale.
    1: "#D55E00",  # vermillion, actionable focus
    2: "#E69F00",  # orange
    3: "#F0E442",  # yellow, the London baseline
    4: "#56B4E9",  # sky blue, quieter areas (CB-safe alternative to green)
}
LONDON_FEATURES = {
    "severity_weighted_count": "Severity-weighted crime",
    "ntl_mean_radiance": "Night-time lights (VIIRS)",
    "employment_deprivation": "Employment deprivation",
    "resolution_rate": "Resolution rate",
}

ENGLAND_TIER_LABEL = {1: "Highest", 2: "High", 3: "Moderate", 4: "Low"}
ENGLAND_TIER_COLOR = {
    # 4-class ColorBrewer Reds, a colourblind-safe sequential ramp.
    1: "#99000D",  # deep red
    2: "#FB6A4A",
    3: "#FCAE91",
    4: "#FEE5D9",  # pale
}

ENGLAND_FEATURES = {
    "ntl_mean_radiance": "Night-time lights (VIIRS)",
    "employment_deprivation": "Employment deprivation",
    "resolution_rate": "Resolution rate",
}

# One-line plain-language gloss per score driver, shown under the detail
# panel's feature table so a reader does not need the Method tab to know what
# each driver means. Keyed by the model's column name.
FEATURE_GLOSS = {
    "severity_weighted_count": "Severity-weighted crime: counts weighted by offence harm",
    "ntl_mean_radiance": "Night-time lights: satellite brightness, a night-activity proxy",
    "employment_deprivation": "Employment deprivation: the IMD 2025 employment domain",
    "resolution_rate": "Resolution rate: share of crimes with a charged or positive outcome",
}

GEOS: dict[str, dict] = {
    "London": {
        "key": "london",
        "scope": "London",
        "caption": ("Proof of concept, not a deployment tool. A planning aid "
                    "that scores demand from 36 months of recorded crime across "
                    "all 4,994 London neighbourhoods."),
        "clusters_name": "london_final_tiers.parquet",
        "clusters_dirs": [ROOT / "team_model", ROOT.parent / "phase5", ROOT / "phase5"],
        "tier_label": LONDON_TIER_LABEL,
        "tier_color": LONDON_TIER_COLOR,
        "tier1_note": ("calibrated against next-year top-15% demand. Contains "
                       "25.3% of London LSOAs and captured 81.7% of the "
                       "realised future hotspots in validation"),
        "features": LONDON_FEATURES,
        # Headline figures surfaced in the Method tab. Kept here as the single
        # source of truth so the tab body never re-types a number that could
        # silently desync from the model after a pipeline re-run.
        "auc": "0.92",
        "tier1_pct": "25.3%",
        "tier1_recall": "81.7%",
        "weights": {
            "severity_weighted_count": "0.77",
            "ntl_mean_radiance": "0.13",
            "employment_deprivation": "0.08",
            "resolution_rate": "0.03",
        },
        "map_mode": "lsoa",
        "geojson": LSOA_GEOJSON,
        "area_word": "borough",
        "area_word_plural": "boroughs",
        "n_areas": 4994,
        "has_allocation": True,
        "has_weather": True,
        "has_overlay": True,
        "has_stations": True,
        "has_crime_weights": True,
        "model_blurb": ("Negative-Binomial + Random-Forest weights over four "
                        "validated features: severity-weighted crime (0.77), "
                        "VIIRS night-time lights (0.13), employment deprivation "
                        "(0.08), resolution rate (0.03). Stop-and-search was "
                        "excluded from the index on ethical grounds (risk of "
                        "compounding enforcement bias); seasonality is kept as "
                        "diagnostic context and in the dynamic layer. Tier "
                        "boundaries are supervised, calibrated on a 24-month "
                        "fit against the next 12 months' realised hotspots "
                        "(AUC 0.92)."),
    },
    "England": {
        "key": "england",
        "scope": "England",
        "caption": ("National structural model: England LSOAs only (the data "
                    "has no Wales/Scotland/NI). Unlike the London index it "
                    "deliberately excludes crime volume, ranking areas by "
                    "underlying structural pressure instead; the event overlay "
                    "and allocation panels are London-only and switch off here. "
                    "Scores are normalised within each scope, so London and "
                    "England figures aren't directly comparable."),
        "clusters_name": "england_final_tiers.parquet",
        "clusters_dirs": [ROOT / "england_model", ROOT.parent / "phase5"],
        "tier_label": ENGLAND_TIER_LABEL,
        "tier_color": ENGLAND_TIER_COLOR,
        "tier1_note": ("calibrated against next-year top-15% demand. Contains "
                       "34.8% of England LSOAs and captured 77.2% of the "
                       "realised future hotspots in validation"),
        "features": ENGLAND_FEATURES,
        # Single source of truth for the figures the Method tab displays.
        "auc": "0.83",
        "tier1_pct": "34.8%",
        "tier1_recall": "77.2%",
        "weights": {
            "ntl_mean_radiance": "0.45",
            "resolution_rate": "0.29",
            "employment_deprivation": "0.26",
        },
        "map_mode": "lad",
        "geojson": LAD_GEOJSON_MERGED,
        "area_word": "local authority",
        "area_word_plural": "local authorities",
        "n_areas": 33_755,
        "has_allocation": False,
        "has_weather": False,
        "has_overlay": False,
        "has_stations": False,
        "has_crime_weights": False,
        "model_blurb": ("Structural demand index: Negative-Binomial + "
                        "Random-Forest weights over three nationally available "
                        "features: VIIRS night-time lights (0.45), resolution "
                        "rate (0.29), employment deprivation (0.26). Crime "
                        "volume is deliberately left out, so the index reads "
                        "as structural pressure independent of recorded crime. "
                        "Tier boundaries are supervised, calibrated against "
                        "the next 12 months' realised hotspots (AUC 0.83)."),
    },
}

# The England model labels four areas with their April-2023 merged-authority
# names, but the Dec-2022 boundary file still carries the predecessor districts.
# Map each predecessor polygon onto its parent so the choropleth fills instead
# of leaving white gaps across the north and south-west.
LAD_PARENT: dict[str, str] = {
    **{d: "Cumberland" for d in ("Allerdale", "Carlisle", "Copeland")},
    **{d: "Westmorland and Furness" for d in ("Barrow-in-Furness", "Eden",
                                              "South Lakeland")},
    **{d: "North Yorkshire" for d in ("Craven", "Hambleton", "Harrogate",
                                      "Richmondshire", "Ryedale", "Scarborough",
                                      "Selby")},
    **{d: "Somerset" for d in ("Mendip", "Sedgemoor",
                               "Somerset West and Taunton", "South Somerset")},
}

# Officer-tunable strengths for the dynamic layer (wires the
# phase7/dynamic_factor_config.json). Order: highest-confidence first. Each
# strength scales that factor's deviation from 1.0 (0 = off, 1 = literature/data
# default, 2 = double).
DYN_FACTORS = [
    ("weekend", "Weekend (Fri-Sun)", "medium-high"),
    ("nightlife_alcohol", "Fri/Sat nightlife", "medium-high"),
    ("temperature", "Temperature", "medium"),
    ("football", "Football match day", "medium"),
    ("events", "Major events", "planning"),
    ("holiday", "Bank holiday", "low-medium"),
]

COVERAGE_THRESHOLD_KM = 1.5

# Default per-crime-type importance weights for the SSA5 slider block (London).
# Defaults are loosely calibrated against the Cambridge Crime Harm Index
# (Sherman, Neyroud & Neyroud, 2016) rescaled to 0-100, so violence and
# robbery sit at the top and low-harm categories at the bottom. The user
# can move any slider to override this on the fly. Order is top-of-volume
# first so the sidebar reads naturally.
CRIME_WEIGHT_DEFAULTS: dict[str, int] = {
    "Violence and sexual offences": 95,
    "Robbery": 70,
    "Burglary": 55,
    "Possession of weapons": 60,
    "Drugs": 35,
    "Anti-social behaviour": 30,
    "Vehicle crime": 25,
    "Public order": 25,
    "Criminal damage and arson": 25,
    "Theft from the person": 30,
    "Shoplifting": 15,
    "Other theft": 15,
    "Bicycle theft": 10,
    "Other crime": 5,
}

# Open-Meteo current-conditions endpoint, no API key. Cached for one
# hour on the Streamlit side. London only.
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=51.5074&longitude=-0.1278"
    "&current=temperature_2m,cloud_cover,precipitation,is_day,wind_speed_10m"
    "&daily=sunrise,sunset"
    "&timezone=Europe%2FLondon"
)


# ----- loaders -----

@st.cache_data(show_spinner="Loading boundaries…")
def load_geojson(path_str: str) -> dict | None:
    p = Path(path_str)
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner="Loading police stations…")
def load_police_geojson() -> dict:
    return json.loads(POLICE_GEOJSON.read_text())


@st.cache_data(show_spinner="Loading model…")
def load_clusters(geo_key: str) -> pd.DataFrame:
    geo = GEOS[geo_key]
    path = _resolve(geo["clusters_name"], geo["clusters_dirs"])
    df = pd.read_parquet(path)
    df = df.rename(columns={"lsoa21cd": "lsoa", "lsoa21nm": "lsoa_name",
                            "lad22nm": "area"})
    # Override the parquet's tier_label (if any) with our local dict so the
    # dashboard can relabel without re-shipping the team's parquet.
    df["tier_label"] = df["tier"].map(geo["tier_label"])
    return df


@st.cache_data(show_spinner="Loading per-crime-type counts…")
def load_crime_pivot() -> pd.DataFrame:
    """4,994 LSOAs x 14 crime types, total counts 2023-03 to 2026-02 (London).

    Built once by 11_dashboard_extras.py. Used by the crime-importance
    slider block to rebuild the demand score live."""
    name = "lsoa_crime_by_type_london.parquet"
    candidates = [
        ROOT / "out" / name,         # local dev layout
        ROOT / "phase5" / name,      # HF Space layout (src/phase5/)
        ROOT.parent / "phase5" / name,
        ROOT / "team_model" / name,
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"{name} not found in {[str(p) for p in candidates]}. "
        "Run `uv run python 11_dashboard_extras.py` to build it."
    )


@st.cache_data(ttl=3600, show_spinner="Checking London weather…")
def fetch_london_weather() -> dict | None:
    """Current London weather from Open-Meteo. None if the fetch fails
    (no network on Streamlit Cloud's healthz, etc.)."""
    try:
        import requests
        r = requests.get(WEATHER_URL, timeout=4)
        r.raise_for_status()
        payload = r.json()
        cur = payload.get("current", {})
        daily = payload.get("daily", {})
        return {
            "temp_c": cur.get("temperature_2m"),
            "cloud_cover_pct": cur.get("cloud_cover"),
            "precipitation_mm": cur.get("precipitation"),
            "is_day": bool(cur.get("is_day")),
            "wind_kmh": cur.get("wind_speed_10m"),
            "sunrise": (daily.get("sunrise") or [None])[0],
            "sunset": (daily.get("sunset") or [None])[0],
        }
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner="Fetching the day's weather…")
def fetch_daily_weather(iso_date: str) -> dict | None:
    """London daily-mean temperature (°C) and mean cloud cover (%) for one date.

    Open-Meteo archive for past dates (the whole 2025-26 season is historical),
    forecast for the next ~16 days. Returns None if neither covers the date, so
    the caller falls back to live current conditions.
    """
    try:
        import requests
        d = pd.to_datetime(iso_date).date()
        today = pd.Timestamp.now().date()
        base = ("https://archive-api.open-meteo.com/v1/archive" if d < today
                else "https://api.open-meteo.com/v1/forecast")
        url = (f"{base}?latitude=51.5074&longitude=-0.1278"
               f"&start_date={iso_date}&end_date={iso_date}"
               f"&daily=temperature_2m_mean&hourly=cloud_cover"
               f"&timezone=Europe%2FLondon")
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        temp = (j.get("daily", {}).get("temperature_2m_mean") or [None])[0]
        cc_h = [c for c in (j.get("hourly", {}).get("cloud_cover") or [])
                if c is not None]
        cc = float(np.mean(cc_h)) if cc_h else None
        if temp is None and cc is None:
            return None
        return {"temp_c": temp, "cloud_cover_pct": cc, "date": iso_date}
    except Exception:
        return None


def _sky_label(pct: float | None) -> str:
    """Cloud-cover percentage to a one-word sky description."""
    if pct is None:
        return "n/a"
    if pct < 20:
        return "Clear"
    if pct < 50:
        return "Partly cloudy"
    if pct < 85:
        return "Mostly cloudy"
    return "Overcast"


def _polygon_centroid(geom: dict) -> tuple[float | None, float | None]:
    """Mean of the outer ring vertices. Good enough at LSOA scale."""
    g_type = geom.get("type")
    if g_type == "Polygon":
        ring = geom["coordinates"][0]
    elif g_type == "MultiPolygon":
        # Pick the largest sub-polygon's outer ring.
        ring = max(geom["coordinates"], key=lambda p: len(p[0]))[0]
    else:
        return None, None
    arr = np.asarray(ring, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@st.cache_data(show_spinner="Computing station distances…")
def load_station_distances() -> pd.DataFrame:
    """LSOA to nearest police station, in km, with station name (London).

    Pure-numpy haversine, no geopandas: keeps the Streamlit Cloud
    memory footprint well under the 1 GB free-tier limit.
    """
    gj = load_geojson(str(LSOA_GEOJSON))
    feats = gj["features"] if gj else []
    codes: list[str] = []
    cent_ll: list[tuple[float, float]] = []  # lat, lon
    for f in feats:
        code = f["properties"].get("LSOA21CD")
        lon, lat = _polygon_centroid(f["geometry"])
        if code is None or lon is None:
            continue
        codes.append(code)
        cent_ll.append((lat, lon))
    cent = np.array(cent_ll)  # [N, 2]

    stations_gj = load_police_geojson()
    sta_records = []
    sta_names: list[str] = []
    for f in stations_gj["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        sta_records.append((lat, lon))
        sta_names.append(f["properties"].get("name", "Police"))
    sta = np.array(sta_records)  # [M, 2]

    # Vectorised haversine: N centroids x M stations.
    r_km = 6371.0
    lat1 = np.radians(cent[:, 0])[:, None]
    lat2 = np.radians(sta[:, 0])[None, :]
    dlat = lat2 - lat1
    dlon = np.radians(sta[:, 1][None, :] - cent[:, 1][:, None])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dists = 2 * r_km * np.arcsin(np.sqrt(a))

    nearest_idx = np.argmin(dists, axis=1)
    nearest_km = dists[np.arange(len(codes)), nearest_idx]
    nearest_names = [sta_names[i] for i in nearest_idx]
    return pd.DataFrame({
        "lsoa": codes,
        "dist_km": nearest_km,
        "station_name": nearest_names,
    })


# ----- allocation / scoring -----

# Met Basic Command Units and the boroughs each one owns (Met FOI, April 2024,
# "Local Policing and Crime: MPS borough codes"). The City of London is policed
# by a separate force, not a Met BCU, so it is deliberately absent here. Its
# LSOAs get no allocation proposal.
BCU_TO_BOROUGHS = {
    "AS - Central South": ["Lambeth", "Southwark"],
    "AW - Central West":  ["Hammersmith and Fulham", "Kensington and Chelsea", "Westminster"],
    "CE - Central East":  ["Hackney", "Tower Hamlets"],
    "CN - Central North": ["Camden", "Islington"],
    "EA - East Area":     ["Barking and Dagenham", "Havering", "Redbridge"],
    "NA - North Area":    ["Enfield", "Haringey"],
    "NE - North East":    ["Newham", "Waltham Forest"],
    "NW - North West":    ["Barnet", "Brent", "Harrow"],
    "SE - South East":    ["Bexley", "Greenwich", "Lewisham"],
    "SN - South Area":    ["Bromley", "Croydon", "Sutton"],
    "SW - South West":    ["Kingston upon Thames", "Merton", "Richmond upon Thames", "Wandsworth"],
    "WA - West Area":     ["Ealing", "Hillingdon", "Hounslow"],
}
BORO_TO_BCU = {b: bcu for bcu, bs in BCU_TO_BOROUGHS.items() for b in bs}

# The 33 Greater London local authorities (32 boroughs + the City of London),
# collapsed to one "Greater London" region on the national (England) map.
LONDON_LADS = set(BORO_TO_BCU) | {"City of London"}


def collapse_london(area: pd.Series) -> pd.Series:
    """Map any Greater London borough to 'Greater London', leave others as-is."""
    return area.where(~area.isin(LONDON_LADS), GREATER_LONDON)

# Real frontline strength per BCU = police-officer FTE + PCSO FTE, from the MPS
# workforce report dated 2026-03-31 (09_efficiency_features.py writes these to
# out/efficiency_bcu.parquet). Hardcoded as a fallback so the deployed Space
# never depends on that parquet being bundled; load_bcu_fte() prefers the
# parquet when present, so re-running 09 on newer data flows straight through.
BCU_FTE_FALLBACK = {
    "AS - Central South": 1515.49, "AW - Central West": 2066.65,
    "CE - Central East":  1409.17, "CN - Central North": 1355.24,
    "EA - East Area":     1541.05, "NA - North Area":    1329.70,
    "NE - North East":    1366.96, "NW - North West":    1584.55,
    "SE - South East":    1700.78, "SN - South Area":    1628.33,
    "SW - South West":    1542.89, "WA - West Area":     1776.62,
}


@st.cache_data
def load_bcu_fte() -> dict:
    """BCU frontline FTE, preferring out/efficiency_bcu.parquet, else fallback."""
    name = "efficiency_bcu.parquet"
    for p in (ROOT / "out" / name, ROOT / "phase5" / name,
              ROOT.parent / "phase5" / name, ROOT / "team_model" / name):
        if p.exists():
            t = pd.read_parquet(p)
            return dict(zip(t["bcu_long"], t["frontline_fte"]))
    return dict(BCU_FTE_FALLBACK)


def allocate_officers_bcu(df: pd.DataFrame,
                          score_col: str = "display_score") -> pd.DataFrame:
    """Officers per London LSOA from the real frontline strength of the BCU it
    sits in, split *within* that BCU in proportion to the demand score.

    Each LSOA's borough maps to one of the 12 Met BCUs; that BCU's published
    frontline FTE is shared across its LSOAs by demand, and compared against an
    even split of the *same* BCU's strength. So `officers_delta` reads as
    "officers above or below an equal share of this command unit": the tool
    never moves officers between BCUs.

    City of London LSOAs map to no BCU (separate force) and stay NaN, which the
    UI surfaces as "no proposal".
    """
    df = df.copy()
    bcu_fte = load_bcu_fte()
    df["bcu"] = df["area"].map(BORO_TO_BCU)
    df["bcu_fte"] = df["bcu"].map(bcu_fte)
    df["officers_proposed"] = np.nan
    df["officers_equal"] = np.nan
    for bcu, grp in df.groupby("bcu"):  # NaN bcu (City of London) is dropped
        pool = bcu_fte.get(bcu)
        w = grp[score_col].clip(lower=0)
        if pool is None or not np.isfinite(pool) or w.sum() <= 0:
            continue
        df.loc[grp.index, "officers_proposed"] = pool * w / w.sum()
        df.loc[grp.index, "officers_equal"] = pool / len(grp)
    df["officers_delta"] = df["officers_proposed"] - df["officers_equal"]
    return df


def _headroom_softcap(base_score: pd.Series, uplift: pd.Series,
                      full: float = 100.0) -> pd.Series:
    """Merge the dynamic uplift onto the 0-100 score without a flat plateau at
    `full`. Positive uplift consumes the REMAINING room to `full` asymptotically:

        disp = full − head · exp(−uplift / head),   head = full − base.

    This is exact identity at uplift==0 (disp == base), strictly increasing in
    uplift at fixed base (rank preserved, no wall of exact ties at `full`), and
    bounded below `full`. Negative uplift (cooling) stays linear toward 0. A
    hard `clip(base + uplift, 0, 100)` ceiling would pin every heavily-boosted
    LSOA to exactly 100 and flatten the map and the within-BCU officer split on
    the busiest days, so this softcap is used instead."""
    b = base_score.to_numpy(dtype=float)
    u = uplift.reindex(base_score.index).fillna(0.0).to_numpy(dtype=float)
    head = np.clip(full - b, 1e-9, None)
    pos = u > 0.0
    out = b + u                                  # zero / cooling branch: linear
    with np.errstate(over="ignore"):             # exp on the u<0 mask can overflow; np.where discards it
        out = np.where(pos, full - head * np.exp(-u / head), out)
    out = np.clip(out, 0.0, full)
    return pd.Series(out, index=base_score.index)


def compute_user_score(panel: pd.DataFrame, crime_pivot: pd.DataFrame,
                       weights: dict[str, float]) -> pd.Series:
    """User-defined demand score per LSOA from per-crime importance weights.

    For each LSOA, score = sum_t (weight_t * count_t), then min-max
    scaled to 0..100 so it lines up with the team's risk_score_scaled.
    """
    relevant = [c for c in weights if c in crime_pivot.columns]
    if not relevant or sum(weights[c] for c in relevant) == 0:
        # Fall back to team score on the merged frame.
        return panel.set_index("lsoa")["risk_score_scaled"]

    arr = crime_pivot[relevant].to_numpy(dtype=float)
    w = np.array([weights[c] for c in relevant], dtype=float)
    raw = arr @ w
    # Percentile rank, then linearly compressed onto the team score's
    # observed range so the colour ramp behaves the same in both modes.
    raw_series = pd.Series(raw, index=crime_pivot["lsoa21cd"].to_numpy())
    pct = raw_series.rank(pct=True, method="average") * 100.0
    team = panel.set_index("lsoa")["risk_score_scaled"]
    team_lo, team_hi = float(team.min()), float(team.max())
    if team_hi > team_lo:
        scaled = team_lo + (pct / 100.0) * (team_hi - team_lo)
    else:
        scaled = pct
    return pd.Series(scaled, index=raw_series.index)


def blend_scores(team: pd.Series, user: pd.Series, mode: str,
                 blend_pct: int) -> pd.Series:
    """Combine the team score and the user-weighted score per the mode."""
    aligned_user = user.reindex(team.index).fillna(team)
    if mode == "Team model":
        return team
    if mode == "My weights":
        return aligned_user
    a = blend_pct / 100.0
    return (1 - a) * team + a * aligned_user


def retier_from_score(score: pd.Series, team_tier: pd.Series,
                      static_score: pd.Series,
                      pin_tier: int | None = None) -> pd.Series:
    """Re-tier `score`, anchored to the static assignment by rank position.

    The supervised tiers' score ranges OVERLAP on the displayed 36-month
    score (their cut-points were calibrated on the 24-month training score).
    Sorting tiers by mean score and filling tier-sized blocks top-down does
    NOT reproduce them when score == static_score (it moved ~6% of London
    LSOAs with the overlay merely switched on). Instead, read
    the static tiers off in static-score order to get the tier at each rank
    position, then hand the LSOA at position i of the NEW ordering that same
    position-i tier. Properties: exact identity when score == static_score;
    a uniform citywide factor (e.g. a weekend multiplier) leaves the ordering
    (and so the tiers) unchanged; per-LSOA factors (nightlife, match day)
    reshuffle membership while preserving every tier's size. That membership
    shift is the dynamic-patrol signal.

    `pin_tier` optionally freezes one tier's members in place (a k-means
    scheme can pin its zero-activity outlier tier). The supervised tiers have
    no such tier, so callers pass None and every LSOA is re-rankable."""
    new = team_tier.astype(int).copy()
    mask = (team_tier != pin_tier) if pin_tier is not None \
        else pd.Series(True, index=team_tier.index)
    if not mask.any():
        return new
    idx = team_tier.index[mask]
    s_old = static_score.reindex(idx).to_numpy(dtype=float)
    # An LSOA with no new score keeps its static position (no information).
    s_new = score.reindex(idx).to_numpy(dtype=float)
    s_new = np.where(np.isnan(s_new), s_old, s_new)
    t_old = team_tier[mask].to_numpy(dtype=int)
    order_old = np.argsort(-s_old, kind="stable")  # positions, highest first
    order_new = np.argsort(-s_new, kind="stable")
    seq = t_old[order_old]                         # tier held at each position
    out = np.empty(len(idx), dtype=int)
    out[order_new] = seq
    new.loc[idx] = out
    return new


def make_recommendation(row: pd.Series, geo: dict, panel: pd.DataFrame,
                        dist_km: float | None) -> str:
    bits: list[str] = []
    delta = row["officers_delta"]
    label = row["tier_label"]
    scope = geo["scope"]

    if pd.isna(row.get("officers_proposed", np.nan)):
        bits.append(f"**{label}.** City of London Police area, a separate force "
                    "outside the Met's command units, so this tool makes no "
                    "officer-allocation proposal here.")
    elif float(row.get("crime_count", 1) or 0) <= 0:
        bits.append(f"**{label}.** Effectively no recorded activity in this "
                    "LSOA (typically industrial, water or transit land). Not a "
                    "target for patrol allocation.")
    elif delta > 0.5:
        bits.append(f"**{label}.** Add about {abs(delta):.0f} officers above an "
                    "even split of the BCU's strength.")
    elif delta < -0.5:
        bits.append(f"**{label}.** Could free up about {abs(delta):.0f} officers "
                    "for elsewhere in the BCU.")
    else:
        bits.append(f"**{label}.** Around the BCU baseline "
                    f"({delta:+.1f} officers).")

    if geo["has_stations"]:
        has_dist = dist_km is not None and not pd.isna(dist_km)
        if has_dist:
            if dist_km > COVERAGE_THRESHOLD_KM and int(row["tier"]) == 1:
                bits.append(f"Coverage gap. Closest station is "
                            f"**{dist_km:.1f} km** away.")
            else:
                bits.append(f"Closest station {dist_km:.1f} km away.")
        else:
            bits.append("No boundary geometry found for this neighbourhood.")

    # Dominant feature relative to the area average, as a ratio rather
    # than a z-score so it reads cleanly in a briefing.
    best_ratio, best_name = 1.0, None
    for col in geo["features"]:
        mu = panel[col].mean()
        if mu and not np.isnan(mu) and mu > 0:
            ratio = row[col] / mu
            if ratio > best_ratio:
                best_ratio, best_name = ratio, geo["features"][col]
    if best_name and best_ratio >= 1.5:
        bits.append(f"Main driver: **{best_name}** "
                    f"({best_ratio:.0f}× {scope} average).")

    return "  \n".join(bits)


# ----- session state -----

# Deferred reset: the Reset button sets _do_reset and reruns. We clear every
# widget's stored state here, before any widget is instantiated this run (so it
# is safe), keeping only the chosen scope. The init loop below then re-creates
# the baseline keys, and each widget falls back to its default value.
if st.session_state.pop("_do_reset", False):
    for _rk in [k for k in list(st.session_state.keys()) if k != "geo_key"]:
        del st.session_state[_rk]

for _k in ("selected_lsoa", "selected_lad", "last_consumed_click_sig",
           "last_consumed_lad_sig"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ----- UI -----

_FAVICON = ASSETS / "favicon.png"
st.set_page_config(page_title="Metropolitan Police demand dashboard",
                   page_icon=str(_FAVICON) if _FAVICON.exists() else None,
                   layout="wide")

if STYLE_CSS.exists():
    st.markdown(f"<style>{STYLE_CSS.read_text()}</style>", unsafe_allow_html=True)

# ----- sidebar: geography toggle FIRST (so the header can react to it) -----

with st.sidebar:
    st.header("Controls")
    st.subheader("Area")
    geo_key = st.segmented_control(
        "Scope",
        options=list(GEOS.keys()),
        default="London",
        key="geo_key",
        help="London: the team's four-feature LSOA index. "
             "England: the national structural index, mapped by local "
             "authority.",
    )
    if not geo_key:
        geo_key = "London"
    if st.button("Reset filters", width="stretch",
                 help="Clear all filters, sliders and selections "
                      "(keeps the current scope)."):
        st.session_state["_do_reset"] = True
        st.rerun()

GEO = GEOS[geo_key]
TIER_LABEL = GEO["tier_label"]
TIER_COLOR = GEO["tier_color"]
FEATURES = GEO["features"]
LOWEST_TIER = max(TIER_LABEL)  # lowest-demand tier number (4, "Low")

# Deferred tier presets (the "Top tier only" / "Hide lowest tier" buttons set
# this and rerun): applied before the tier checkboxes instantiate, so we can set
# their state without Streamlit's "modified after instantiated" error.
_preset = st.session_state.pop("_tier_preset", None)
if _preset:
    for _t in TIER_LABEL:
        _ckey = f"tier_show_{geo_key}_{_t}"
        if _preset == "top":
            st.session_state[_ckey] = (_t == 1)
        elif _preset == "hide_lowest":
            st.session_state[_ckey] = (_t != LOWEST_TIER)

# ----- load active model -----

phase5 = load_clusters(geo_key)
areas_all = sorted(phase5["area"].dropna().unique())
if GEO["map_mode"] == "lad":
    # The England map shows London as one region, so the filter and counts treat
    # it as one "Greater London" entry instead of 33 separate boroughs.
    areas_all = sorted(collapse_london(pd.Series(areas_all)).unique())
dist_df = load_station_distances() if GEO["has_stations"] else None
weather = fetch_london_weather() if GEO["has_weather"] else None


def _ribbon_html(w: dict | None) -> str:
    """Glanceable one-line live-conditions ribbon for the header."""
    if not w:
        return "<span class='ribbon'>live conditions offline</span>"
    bits: list[str] = []
    if w.get("temp_c") is not None:
        bits.append(f"<b>{w['temp_c']:.1f}&deg;C</b>")
    sky = _sky_label(w.get("cloud_cover_pct"))
    if sky != "n/a":
        bits.append(sky.lower())
    bits.append("<b>Day</b>" if w.get("is_day") else "<b>Night</b>")
    if w.get("is_day") and w.get("sunset"):
        bits.append(f"sunset {str(w['sunset'])[-5:]}")
    elif (not w.get("is_day")) and w.get("sunrise"):
        bits.append(f"sunrise {str(w['sunrise'])[-5:]}")
    return ("<span class='ribbon'>"
            + "<span class='dot'>&middot;</span>".join(bits)
            + "</span>")


_n_tiers = len(TIER_LABEL)
_head_right = _ribbon_html(weather) if GEO["has_weather"] else (
    f"<span class='ribbon'>{GEO['scope']} &middot; "
    f"{len(areas_all)} {GEO['area_word_plural']}</span>"
)
_head_title = ("Metropolitan Police" if geo_key == "London"
               else f"{GEO['scope']} policing demand")
_head_sub = (
    f"Policing demand across {GEO['n_areas']:,} London neighbourhoods, "
    f"in {_n_tiers} priority tiers."
    if geo_key == "London"
    else f"{GEO['n_areas']:,} neighbourhoods scored, in {_n_tiers} priority tiers."
)
st.markdown(
    "<div class='app-head'><div>"
    f"<div class='title'>{_head_title}</div>"
    f"<div class='sub'>{_head_sub}</div></div>"
    + _head_right
    + "</div>",
    unsafe_allow_html=True,
)
if GEO["caption"]:
    st.caption(GEO["caption"])

# ----- rest of sidebar -----

with st.sidebar:
    area_sel = st.multiselect(
        f"Filter to {GEO['area_word_plural']}",
        areas_all,
        default=[],
        key=f"area_sel_{geo_key}",
        placeholder=f"All {GEO['scope']}",
    )

    st.subheader("Tiers shown")
    tier_opts = list(TIER_LABEL.keys())
    # One checkbox per tier, so which tiers are visible is obvious at a glance
    # instead of hidden inside a dropdown.
    visible_tiers = [
        t for t in tier_opts
        if st.checkbox(f"{t}. {TIER_LABEL[t]}", value=True,
                       key=f"tier_show_{geo_key}_{t}")
    ]
    if not visible_tiers:
        st.caption("No tiers ticked, showing all.")
    qa, qb = st.columns(2)
    if qa.button("Top tier only", width="stretch"):
        st.session_state["_tier_preset"] = "top"
        st.rerun()
    if qb.button("Hide lowest tier", width="stretch",
                  help=f"Hide Tier {LOWEST_TIER}, the lowest-demand "
                       "neighbourhoods, so the map foregrounds where "
                       "resources are contested."):
        st.session_state["_tier_preset"] = "hide_lowest"
        st.rerun()

    weight_mode = "Team model"
    blend_pct = 50
    crime_weights = dict(CRIME_WEIGHT_DEFAULTS)
    if GEO["has_crime_weights"]:
        st.subheader("Scoring weights")
        weight_mode = st.radio(
            "Which weights build the risk score?",
            ["Team model", "My weights", "Blend"],
            index=0, horizontal=True,
            help=("Team model: the group's validated risk index. "
                  "My weights: re-rank LSOAs by how much each crime type "
                  "matters to you. Blend: mix the two."),
            key="weight_mode",
        )
        blend_pct = st.slider(
            "Blend toward my weights", 0, 100, 50, step=5,
            disabled=(weight_mode != "Blend"),
            help="0 = pure team model, 100 = pure user weights.",
        )
        with st.popover("Per crime-type weights", width="stretch"):
            st.caption(
                "0 = ignore this crime type, 100 = weight it most heavily. "
                "Defaults track the Cambridge Crime Harm Index."
            )
            if st.button("Reset to Harm Index defaults", width="stretch",
                         help="Restore the Cambridge Crime Harm Index defaults."):
                for k, v in CRIME_WEIGHT_DEFAULTS.items():
                    st.session_state[f"w_{k}"] = v
                st.rerun()
            for crime, default in CRIME_WEIGHT_DEFAULTS.items():
                key = f"w_{crime}"
                crime_weights[crime] = st.slider(
                    crime, 0, 100, st.session_state.get(key, default),
                    step=5, key=key,
                )

    st.subheader("Map display")
    show_stations = (st.checkbox("Police stations", value=True)
                     if GEO["has_stations"] else False)
    show_top_hotspots = st.checkbox("Highlight top 10", value=True)
    # Borderline-case filter keys off the clustering silhouette; the
    # supervised tiers don't carry one, so the control only appears when the
    # staged parquet has the column.
    confident_only = False
    if "silhouette" in phase5.columns:
        confident_only = st.checkbox(
            "Hide tier-borderline cases", value=False,
            help="Hides neighbourhoods that sit between two tiers. "
                 "Use this if you only want decisive cases.",
            key=f"confident_{geo_key}",
        )

    overlay_on, overlay_date, is_night = False, None, True
    factor_strengths: dict[str, float] = {}
    _otabs = None
    if GEO["has_overlay"]:
        st.subheader("Dynamic day")
        _otabs = get_overlay_tables()
        overlay_on = st.toggle(
            "Dynamic factor layer", value=True,
            disabled=(_otabs is None),
            help="Master switch. ON: the map, tiers and officer split use "
                 "risk_now = static index × date-specific factors (temperature, "
                 "weekend, nightlife, football, events, bank holiday), capped "
                 "0.5-2.5×. OFF: the validated static 36-month index. The "
                 "validated model itself is never changed either way.",
        )
        if _otabs is None:
            st.caption("Overlay data not shipped with this build.")
        elif overlay_on:
            _today = pd.Timestamp.now().date()
            _lo = pd.Timestamp("2025-08-01").date()
            _hi = pd.Timestamp("2026-12-31").date()
            _default = min(max(_today, _lo), _hi)
            overlay_date = st.date_input(
                "Planning date", value=_default, min_value=_lo, max_value=_hi,
                help="Pick the day you're planning for. Loaded fixtures cover "
                     "the 2025-26 season (Aug 2025 to May 2026).",
            )
            is_night = st.toggle(
                "Evening / night shift", value=True,
                help="Nightlife only lifts demand on Friday/Saturday evenings. "
                     "Turn off to plan a daytime shift.",
            )
            with st.popover("Factor strengths", width="stretch"):
                # Each factor's calibrated default (the value applied at
                # strength 1.0). Weekend/football come from the S&S calibration
                # in PARAMS so the display matches what's actually applied.
                _P = getattr(_overlay, "PARAMS", {}) if _overlay else {}
                _wkv = 1 + float(_P.get("weekend_pct", 12.0)) / 100
                _fbv = 1 + float(_P.get("football_pct", 22.0)) / 100
                _DFLT = {
                    "weekend": f"Fri/Sat ×{_wkv:.2f}, Sun ×1.05",
                    "nightlife_alcohol": "up to ×1.50 total (absorbs weekend)",
                    "temperature": "+1.2%/°C vs 15°C",
                    "football": f"×{_fbv:.2f} match day",
                    "events": "up to ×1.45",
                    "holiday": "×1.10",
                }
                _SRC = {
                    "weekend": "MPS stop-and-search (Fri/Sat pooled)",
                    "football": "MPS stop-and-search + literature evening/derby",
                    "nightlife_alcohol": "literature; scaled per LSOA by pub/club "
                                         "density; replaces (never stacks on) the "
                                         "weekend factor in those LSOAs",
                    "temperature": "literature; driven by the date's Open-Meteo temp",
                    "events": "planning assumption; scaled by event size",
                    "holiday": "literature",
                }
                st.caption(
                    "Every factor already has a calibrated value. The default "
                    "is shown on each slider, and the live per-date multiplier "
                    "in the Active factors table (Dynamic day tab). The slider "
                    "**scales** that value: 1.0 = use it as calibrated, 0 = off, "
                    "2 = double. Weekend and football come from MPS "
                    "stop-and-search; the rest are literature-informed."
                )
                if st.button("Reset factor strengths", width="stretch"):
                    for fid, _, _ in DYN_FACTORS:
                        st.session_state[f"s_{fid}"] = 1.0
                    st.rerun()
                for fid, label, conf in DYN_FACTORS:
                    factor_strengths[fid] = st.slider(
                        f"{label}: default {_DFLT[fid]}", 0.0, 2.0,
                        st.session_state.get(f"s_{fid}", 1.0),
                        step=0.1, key=f"s_{fid}",
                        help=f"Default at strength 1.0: {_DFLT[fid]}. "
                             f"Source: {_SRC[fid]}. Confidence: {conf}. "
                             "Slider scales it (0 = off, 1 = as calibrated, "
                             "2 = double).",
                    )

# ----- compute -----

team_score = phase5.set_index("lsoa")["risk_score_scaled"]
if weight_mode == "Team model" or not GEO["has_crime_weights"]:
    score_series = team_score
    tier_series = phase5.set_index("lsoa")["tier"]
else:
    crime_pivot = load_crime_pivot()
    user_score = compute_user_score(phase5, crime_pivot, crime_weights)
    score_series = blend_scores(team_score, user_score, weight_mode, blend_pct)
    tier_series = retier_from_score(
        score_series, phase5.set_index("lsoa")["tier"], team_score)

df = phase5.copy()
df["display_score"] = df["lsoa"].map(score_series)
df["tier"] = df["lsoa"].map(tier_series).astype(int)
df["tier_label"] = df["tier"].map(TIER_LABEL)

# ----- dynamic factor layer (London only): risk_now = static × Π factors -----
# Wires Mateus's phase7/dynamic_factor_config.json. compute_overlay_config
# returns uplift = baseline × (combined_factor − 1), so the additive merge here
# reproduces the multiplicative score. Active factors come back in .attrs.
df["dyn_uplift"] = 0.0
df["dyn_reasons"] = ""
dyn_factor_meta: list[dict] = []
dyn_weather_dated = False
dyn_day_weather: dict = {}
dyn_up = dyn_down = 0   # LSOAs that move to a higher / lower tier under the layer
if overlay_on and _otabs is not None and overlay_date is not None and _overlay is not None:
    # Weather for the *planning date* (archive/forecast), not just live current.
    _iso_plan = pd.to_datetime(str(overlay_date)).date().isoformat()
    dyn_day_weather = fetch_daily_weather(_iso_plan) or {}
    dyn_weather_dated = bool(dyn_day_weather)
    _temp = dyn_day_weather.get("temp_c", (weather or {}).get("temp_c"))
    overlay_df = _overlay.compute_overlay_config(
        overlay_date, _otabs,
        baseline=df.set_index("lsoa")["display_score"],
        temp_c=_temp, is_night=is_night,
        strengths=factor_strengths)
    dyn_factor_meta = overlay_df.attrs.get("factors", [])
    if len(overlay_df):
        umap = overlay_df.set_index("lsoa21cd")
        df["dyn_uplift"] = df["lsoa"].map(umap["uplift"]).fillna(0.0)
        df["dyn_reasons"] = df["lsoa"].map(umap["reasons"]).fillna("")
        df["base_pre_overlay"] = df["display_score"].to_numpy()   # true baseline before the merge
        df["display_score"] = _headroom_softcap(
            df.set_index("lsoa")["display_score"],
            df.set_index("lsoa")["dyn_uplift"],
        ).reindex(df["lsoa"]).to_numpy()
        _pre_tier = df["tier"].copy()
        df["tier"] = df["lsoa"].map(
            retier_from_score(df.set_index("lsoa")["display_score"],
                              phase5.set_index("lsoa")["tier"],
                              team_score)).astype(int)
        df["tier_label"] = df["tier"].map(TIER_LABEL)
        # tier 1 = highest risk, so a lower tier number = moved up
        dyn_up = int((df["tier"] < _pre_tier).sum())
        dyn_down = int((df["tier"] > _pre_tier).sum())

# Risk-score span of each tier, computed on the FULL scope before any view
# filter, so the legend can say what e.g. "Highest" means in score terms. The
# tiers are cut on the displayed score (the Option B re-cut, fetch_team_model.py),
# so these bands are contiguous and non-overlapping. The score alone tells you
# the tier. Re-weighting or the dynamic layer re-rank LSOAs but keep the bands
# contiguous.
_tier_ranges = df.groupby("tier")["display_score"].agg(["min", "max"])


def tier_range_txt(t: int) -> str:
    """'62-100'-style risk-score range for tier t, or '' if absent."""
    if t not in _tier_ranges.index:
        return ""
    return (f"{_tier_ranges.loc[t, 'min']:.1f}-"
            f"{_tier_ranges.loc[t, 'max']:.1f}")


# Allocate on the FULL frame so an LSOA's proposed officers don't change when
# the view is filtered. The BCUs and their strength are fixed, not view-relative.
if GEO["has_allocation"]:
    df = allocate_officers_bcu(df, score_col="display_score")

if area_sel:
    # 'Greater London' in the England filter expands to its 33 boroughs, since
    # the underlying rows still carry individual borough names.
    want = set(area_sel)
    if GREATER_LONDON in want:
        want = (want - {GREATER_LONDON}) | LONDON_LADS
    df = df[df["area"].isin(want)]
if visible_tiers:
    df = df[df["tier"].isin(visible_tiers)]
if confident_only and "silhouette" in df.columns:
    df = df[df["silhouette"] >= 0]

if dist_df is not None:
    df = df.merge(dist_df, on="lsoa", how="left")
else:
    df["dist_km"] = np.nan
    df["station_name"] = None

# ----- KPI strip -----

n_total = len(df)
n_hotspot = int((df["tier"] == 1).sum())
top_row = df.nlargest(1, "display_score").iloc[0] if n_total else None
mean_risk = float(df["display_score"].mean()) if n_total else 0.0
officers_reallocated = (
    int(df.loc[df["officers_delta"] > 0, "officers_delta"].sum())
    if GEO["has_allocation"] and "officers_delta" in df.columns else 0)


def _kpi(col, label, value, **kwargs):
    """A metric inside a bordered container, so the KPI row reads as cards."""
    with col:
        with st.container(border=True):
            st.metric(label, value, **kwargs)


# Build the KPI cards dynamically: the allocation card only exists where we have
# real officer data (London), so England drops it and shows one fewer card.
kpis = [
    ("Neighbourhoods in view", f"{n_total:,}", {}),
    (f"{TIER_LABEL[1]}-tier neighbourhoods", f"{n_hotspot:,}",
     {"help": f"Tier 1: {GEO['tier1_note']}."}),
    ("Average risk score", f"{mean_risk:.1f}",
     {"help": ("Risk score averaged across what's currently in view"
               + (f" (highest single score {top_row['display_score']:.0f})."
                  if top_row is not None else "."))}),
]
if GEO["has_allocation"]:
    kpis.append((
        "Officers reallocated (in view)" if area_sel else "Officers reallocated",
        f"{officers_reallocated:,}",
        {"help": "Officers shifted toward higher-demand neighbourhoods within "
                 "each BCU (Basic Command Unit), versus an even split of that "
                 "BCU's own strength. The total reflects the neighbourhoods "
                 "currently in view; a single borough shows only its slice of "
                 "a BCU."}))
if GEO["has_stations"]:
    gap_count = int((df["tier"].isin({1}) & (df["dist_km"] > COVERAGE_THRESHOLD_KM)).sum())
    kpis.append((
        "Coverage gaps", f"{gap_count}",
        {"delta": f">{COVERAGE_THRESHOLD_KM:g} km from a station",
         "delta_color": "off",
         "help": "High-risk neighbourhoods with no station within walking distance."}))
else:
    n_areas_view = int(collapse_london(df["area"]).nunique())
    kpis.append((
        GEO["area_word_plural"].capitalize(), f"{n_areas_view:,}",
        {"help": f"Number of {GEO['area_word_plural']} represented in the current "
                 "view (Greater London counts as one)."}))

for _col, (_label, _value, _kw) in zip(st.columns(len(kpis)), kpis):
    _kpi(_col, _label, _value, **_kw)

# ----- tabs (geography-aware) -----

_tab_titles = ["Map", "Tiers & gaps"]
if GEO["has_overlay"]:
    _tab_titles.append("Dynamic day")
_tab_titles.append("Method")
_tabs = st.tabs(_tab_titles)
tab_map = _tabs[0]
tab_tiers = _tabs[1]
tab_day = _tabs[2] if GEO["has_overlay"] else None
tab_method = _tabs[-1]


# ===== MAP TAB =====

def _detail_panel_lsoa(col, df, phase5, geo, weight_mode):
    """Right-hand detail column for an LSOA-level (London) selection."""
    with col:
        selected = st.session_state.selected_lsoa
        if selected and selected in set(df["lsoa"]):
            row = df[df["lsoa"] == selected].iloc[0]
            with st.container(border=True):
                head_col, clear_col = st.columns([3, 1])
                head_col.subheader(f"{row['lsoa_name']}")
                _bcu = row.get("bcu")
                _bcu_txt = (f"{str(_bcu).split(' - ')[-1]} BCU"
                            if isinstance(_bcu, str) else "City of London Police")
                head_col.markdown(
                    f"<span class='muted'>{row['area']} · {_bcu_txt} · "
                    f"{row['lsoa']}</span>", unsafe_allow_html=True)
                if clear_col.button("Clear", width="stretch"):
                    st.session_state.selected_lsoa = None
                    st.rerun()

                m1, m2 = st.columns(2)
                team_score_val = float(row["risk_score_scaled"])
                m1.metric(
                    "Risk score", f"{row['display_score']:.1f}",
                    delta=(None if weight_mode == "Team model"
                           else f"{row['display_score'] - team_score_val:+.1f} vs team model"),
                    delta_color="off")
                m2.metric("Tier", f"{row['tier']}. {row['tier_label']}")
                m3, m4 = st.columns(2)
                if pd.isna(row.get("officers_proposed")):
                    m3.metric("Proposed officers", "n/a",
                              delta="City of London (separate force)",
                              delta_color="off")
                else:
                    m3.metric("Proposed officers", f"{row['officers_proposed']:.1f}",
                              delta=f"{row['officers_delta']:+.1f} vs even split in BCU",
                              help="Split within this neighbourhood's BCU (Basic "
                                   "Command Unit), the borough group the Met "
                                   "allocates through, in proportion to demand.")
                if geo["has_stations"] and pd.notna(row["dist_km"]):
                    m4.metric("Nearest station", f"{row['dist_km']:.1f} km",
                              delta=row["station_name"] if pd.notna(row["station_name"]) else None,
                              delta_color="off")
                elif geo["has_stations"]:
                    m4.metric("Nearest station", "n/a",
                              delta="No map polygon", delta_color="off")
                # Within-scope rank turns "Tier 1" from a quarter-of-the-city
                # label into a targetable position.
                _rank = int((phase5["risk_score_scaled"]
                             > row["risk_score_scaled"]).sum()) + 1
                st.caption(f"Static risk-score rank: **#{_rank:,}** of "
                           f"{len(phase5):,} in {geo['scope']}.")
                if "silhouette" in row:
                    st.metric("Tier confidence", f"{row['silhouette']:+.2f}",
                              help="How clearly this neighbourhood fits its tier. "
                                   "Negative means the model could place it in a "
                                   "neighbouring tier instead.")

            st.markdown("**Recommended action**")
            st.info(make_recommendation(
                row, geo, phase5,
                row["dist_km"] if pd.notna(row["dist_km"]) else None))

            st.markdown("**What's behind the score**")
            feat_df = pd.DataFrame({
                "Driver": [geo["features"][c] for c in geo["features"]],
                "Value": [float(row[c]) for c in geo["features"]],
                "Area average": [float(phase5[c].mean()) for c in geo["features"]],
            })
            feat_df["× area average"] = (
                feat_df["Value"] / feat_df["Area average"].replace(0, np.nan)
            ).round(2)
            st.dataframe(
                feat_df[["Driver", "Value", "× area average"]].assign(
                    Value=feat_df["Value"].round(1)),
                hide_index=True, width="stretch")
            st.caption(" · ".join(
                FEATURE_GLOSS[c] for c in geo["features"] if c in FEATURE_GLOSS))

            # Dynamic contribution, shown whenever the layer is on (not only
            # when the uplift is positive), so a fresh page load with the
            # default date already explains the dynamic part of the score.
            # display_score is a saturating merge, not base + uplift, so read
            # the stored baseline and report the realised rise.
            if geo["has_overlay"] and overlay_on and overlay_date is not None:
                base_today = float(row.get("base_pre_overlay",
                                           row["display_score"]))
                _shown_rise = float(row["display_score"]) - base_today
                _reasons = [x.strip() for x in
                            str(row.get("dyn_reasons", "")).split(";")
                            if x.strip()]
                if abs(_shown_rise) >= 0.05:
                    _verb = "lifts" if _shown_rise > 0 else "lowers"
                    st.success(
                        f"**Dynamic layer, {overlay_date:%a %d %b %Y}** "
                        f"{_verb} the static score {base_today:.1f} by "
                        f"{_shown_rise:+.1f} to **{row['display_score']:.1f}** "
                        f"({row['tier_label']}).")
                    for r in _reasons:
                        st.markdown(f"- {r}")
                else:
                    st.caption(
                        f"Dynamic layer is on for {overlay_date:%a %d %b %Y}, "
                        "but no date-specific factor moves this neighbourhood, "
                        "so the static and dynamic scores match today.")
        else:
            _leaderboard(df, geo)


def _leaderboard(df, geo):
    # Top-N drill-down: Tier 1 is an allocation envelope (a quarter of the
    # city), so patrol-level targeting happens here, by score rank.
    head, sel = st.columns([3, 1])
    head.subheader("Top hotspots")
    n = sel.selectbox(
        "How many hotspots to list", [10, 25, 50, 100], index=0,
        format_func=lambda x: f"Top {x}", label_visibility="collapsed",
        key=f"topn_{geo['key']}")
    st.markdown(
        "<span class='muted'>Ranked by risk score. Use this list for "
        "neighbourhood-level targeting. Tap a row to inspect, or tap any "
        "neighbourhood on the map.</span>",
        unsafe_allow_html=True)
    topn = df.nlargest(n, "display_score")
    for i, (_, hr) in enumerate(topn.iterrows(), 1):
        label = (f"**{i}.** {hr['lsoa_name']} · {hr['area']} · "
                 f"score {hr['display_score']:.1f} · {hr['tier_label']}")
        if st.button(label, key=f"hot_{hr['lsoa']}", width="stretch"):
            st.session_state.selected_lsoa = hr["lsoa"]
            st.rerun()


def _lad_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Roll the England LSOA frame up to local-authority level for the map.

    Greater London's 33 boroughs are collapsed into a single 'Greater London'
    row so the national map shows it as one region."""
    g = (df.assign(_area=collapse_london(df["area"]))
         .groupby("_area")
         .agg(mean_risk=("display_score", "mean"),
              max_risk=("display_score", "max"),
              n_lsoas=("lsoa", "size"),
              n_top=("tier", lambda s: int((s == 1).sum())))
         .reset_index()
         .rename(columns={"_area": "area"}))
    return g


def _lad_name(props: dict) -> str | None:
    for k in ("LAD22NM", "LAD23NM", "LAD21NM", "lad22nm", "LAD24NM", "name"):
        if k in props and props[k]:
            return props[k]
    return None


def _detail_panel_lad(col, df, lad_agg, geo):
    """Right-hand detail column for a LAD-level (England) selection."""
    with col:
        sel = st.session_state.selected_lad
        if sel and sel in set(lad_agg["area"]):
            is_london = sel == GREATER_LONDON
            arow = lad_agg[lad_agg["area"] == sel].iloc[0]
            sub = df[collapse_london(df["area"]) == sel]
            with st.container(border=True):
                head_col, clear_col = st.columns([3, 1])
                head_col.subheader(sel)
                _subtitle = ("32 boroughs + the City of London"
                             if is_london else geo["scope"])
                head_col.markdown(
                    f"<span class='muted'>{int(arow['n_lsoas'])} LSOAs · "
                    f"{_subtitle}</span>", unsafe_allow_html=True)
                if clear_col.button("Clear", width="stretch"):
                    st.session_state.selected_lad = None
                    st.rerun()
                m1, m2 = st.columns(2)
                m1.metric("Mean risk score", f"{arow['mean_risk']:.1f}",
                          help=f"Highest single LSOA score here: {arow['max_risk']:.0f}.")
                m2.metric("Top-tier LSOAs",
                          f"{int(arow['n_top'])} of {int(arow['n_lsoas'])}")

            if is_london:
                st.info(
                    "**Greater London is our detailed-model region.** On the "
                    "national map it's shown as one area, in its own colour, so "
                    "you can compare it against the rest of England. The score "
                    "here is the England-wide structural model (for a like-for-"
                    "like comparison). Switch to the **London** scope for the "
                    "full four-feature index, per-LSOA tiers and officer "
                    "allocation.")

            st.markdown("**Highest-demand LSOAs here**")
            _cols = ["lsoa_name", "display_score", "tier_label"]
            topn = sub.nlargest(8, "display_score")[_cols].rename(
                columns={"lsoa_name": "Neighbourhood",
                         "display_score": "Score", "tier_label": "Tier"})
            st.dataframe(topn.assign(Score=topn["Score"].round(1)),
                         hide_index=True, width="stretch")
        else:
            st.subheader("Top 10 local authorities")
            st.markdown(
                "<span class='muted'>By mean risk score. Tap a row, or tap "
                "an authority on the map.</span>", unsafe_allow_html=True)
            top10 = lad_agg.nlargest(10, "mean_risk")
            for i, (_, hr) in enumerate(top10.iterrows(), 1):
                label = (f"**{i}.** {hr['area']} · mean {hr['mean_risk']:.1f} · "
                         f"{int(hr['n_lsoas'])} LSOAs")
                if st.button(label, key=f"lad_{hr['area']}", width="stretch"):
                    st.session_state.selected_lad = hr["area"]
                    st.rerun()


with tab_map:
    if GEO["has_overlay"]:
        if overlay_on and _otabs is not None and overlay_date is not None:
            _move = (f"{dyn_up:,} move up a tier" if dyn_up else "no tier moves")
            if dyn_down:
                _move += f", {dyn_down:,} down"
            st.info(
                f"**Dynamic view, {overlay_date:%a %d %b %Y}.** The map, tiers "
                f"and officer split show risk_now = static × factors. {_move} "
                "vs the static index. Breakdown in the Dynamic day tab; switch "
                "back with the sidebar toggle.")
        else:
            st.caption(
                "Showing the **static 36-month index**. Turn on the *Dynamic "
                "factor layer* toggle in the sidebar to overlay a specific date.")

    # Legend: discrete tier swatches for the London LSOA choropleth, a
    # continuous gradient when it's coloured by mean score (England local
    # authorities), so the legend always matches the encoding.
    if GEO["map_mode"] == "lsoa":
        st.markdown(
            "<div class='legend'>"
            + "".join(
                f"<span><span class='sw' style='background:{TIER_COLOR[t]}'></span>"
                f"{t}. {TIER_LABEL[t]} "
                f"<span class='muted'>({tier_range_txt(t)})</span></span>"
                for t in TIER_LABEL)
            + "</div>",
            unsafe_allow_html=True)
        st.caption(f"Brackets show the risk-score range each tier spans on the "
                   f"current score. Tier 1 is an allocation envelope, not a "
                   f"pin-map: {GEO['tier1_note']}. Use the leaderboard's top-N "
                   "for patrol-level targeting.")
    else:
        _grad = (f"linear-gradient(to right, {ENGLAND_TIER_COLOR[4]}, "
                 f"{ENGLAND_TIER_COLOR[2]}, {ENGLAND_TIER_COLOR[1]})")
        st.markdown(
            "<div class='legend'><span>Mean risk score per local authority:</span>"
            "<span class='muted'>lower</span>"
            f"<span style='display:inline-block;width:150px;height:12px;"
            f"border-radius:3px;background:{_grad};margin:0 6px;"
            "border:1px solid rgba(0,0,0,.1)'></span>"
            "<span class='muted'>higher</span>"
            f"<span style='margin-left:14px'><span class='sw' "
            f"style='background:{LONDON_REGION_FILL};border:1px solid "
            f"{LONDON_REGION_LINE}'></span>Greater London "
            "(our detailed model, shown for comparison)</span></div>",
            unsafe_allow_html=True)

    col_map, col_detail = st.columns([2, 1], gap="medium")

    if GEO["map_mode"] == "lsoa":
        # ---------- London: LSOA choropleth ----------
        with col_map:
            risk_lookup = df.set_index("lsoa")["display_score"].to_dict()
            tier_lookup = df.set_index("lsoa")["tier"].to_dict()
            label_lookup = df.set_index("lsoa")["tier_label"].to_dict()
            visible_set = set(df["lsoa"])
            top10 = df.nlargest(10, "display_score")
            top10_set = set(top10["lsoa"])

            m = folium.Map(
                location=[51.509, -0.118], zoom_start=10,
                tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png",
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
            Fullscreen(position="topright", title="Full screen",
                       title_cancel="Exit full screen",
                       force_separate_button=True).add_to(m)

            geojson = load_geojson(str(GEO["geojson"]))
            rendered_features, halo_features = [], []
            for feat in (geojson["features"] if geojson else []):
                code = feat["properties"].get("LSOA21CD")
                if code not in visible_set:
                    continue
                feat["properties"]["risk"] = float(risk_lookup.get(code, 0))
                feat["properties"]["tier"] = int(tier_lookup.get(code, LOWEST_TIER))
                feat["properties"]["tier_label"] = label_lookup.get(code, "n/a")
                rendered_features.append(feat)
                if show_top_hotspots and code in top10_set:
                    halo_features.append(feat)
            rendered_geo = {"type": "FeatureCollection", "features": rendered_features}
            halo_geo = {"type": "FeatureCollection", "features": halo_features}

            if halo_features:
                folium.GeoJson(
                    halo_geo, name="Top hotspots halo",
                    style_function=lambda f: {
                        "fillOpacity": 0, "color": "#FBBF24",
                        "weight": 7, "opacity": 0.55},
                    interactive=False).add_to(m)

            def style_fn(feat):
                tier = feat["properties"].get("tier", LOWEST_TIER)
                fill = TIER_COLOR.get(tier, "#CCCCCC")
                return {"fillColor": fill,
                        "color": "#FFFFFF", "weight": 0, "fillOpacity": 0.92}

            folium.GeoJson(
                rendered_geo, name="LSOAs", style_function=style_fn,
                tooltip=folium.GeoJsonTooltip(
                    fields=["LSOA21CD", "LSOA21NM", "risk", "tier_label"],
                    aliases=["Code", "Neighbourhood", "Score", "Tier"],
                    localize=True, labels=True),
                highlight_function=lambda f: {"weight": 2, "color": "#1F2937"},
            ).add_to(m)

            if show_stations:
                for sf in load_police_geojson()["features"]:
                    lon, lat = sf["geometry"]["coordinates"]
                    folium.CircleMarker(
                        location=[lat, lon], radius=2.5, color="#1E3A8A",
                        weight=1.2, fillColor="#FFFFFF", fillOpacity=0.85,
                        tooltip=sf["properties"].get("name", "Police")).add_to(m)

            map_event = st_folium(m, width=None, height=600,
                                  returned_objects=["last_active_drawing"],
                                  key="risk_map")
            if map_event and map_event.get("last_active_drawing"):
                drawing = map_event["last_active_drawing"]
                clicked = drawing.get("properties", {}).get("LSOA21CD")
                click_sig = json.dumps(drawing, sort_keys=True, default=str)
                if clicked and click_sig != st.session_state.last_consumed_click_sig:
                    st.session_state.last_consumed_click_sig = click_sig
                    st.session_state.selected_lsoa = clicked

        _detail_panel_lsoa(col_detail, df, phase5, GEO, weight_mode)

    else:
        # ---------- England: Local-Authority-District choropleth ----------
        lad_agg = _lad_aggregate(df)
        with col_map:
            geojson = load_geojson(str(GEO["geojson"]))
            if geojson is None:
                st.warning(
                    "England boundary file (`dashboard_assets/england_lad.geojson`) "
                    "isn't present yet, so the national map can't render. The "
                    "leaderboard and tables on the right still work. Run the "
                    "boundary fetch to enable the choropleth.")
            else:
                agg_lookup = lad_agg.set_index("area")
                # Mean scores compress into a narrow band (most authorities
                # score 67-80), so stretch the ramp to the 10th-90th percentile
                # of the visible authorities. Values outside are clamped to the
                # ramp ends. This gives a readable gradient instead of a flat
                # wash of red.
                if len(lad_agg):
                    vmin = float(lad_agg["mean_risk"].quantile(0.10))
                    vmax = float(lad_agg["mean_risk"].quantile(0.90))
                else:
                    vmin, vmax = 0.0, 100.0
                if vmax <= vmin:
                    vmin, vmax = float(lad_agg["mean_risk"].min() if len(lad_agg) else 0.0), vmin + 1.0
                cmap = LinearColormap(
                    [ENGLAND_TIER_COLOR[4], ENGLAND_TIER_COLOR[2], ENGLAND_TIER_COLOR[1]],
                    vmin=vmin, vmax=vmax,
                    caption="Mean LSOA risk score per local authority "
                            "(ramp stretched to the 10th-90th percentile).")

                m = folium.Map(
                    location=[52.8, -1.5], zoom_start=6,
                    tiles="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png",
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>')
                # Frame on England's mainland extent so the map opens filling
                # the panel, not zoomed out over the sea/Continent.
                m.fit_bounds([[49.9, -6.4], [55.85, 1.8]])
                Fullscreen(position="topright", title="Full screen",
                           title_cancel="Exit full screen",
                           force_separate_button=True).add_to(m)

                visible_areas = set(lad_agg["area"])
                top10_lads = set(lad_agg.nlargest(10, "mean_risk")["area"])
                rendered, halo = [], []
                for feat in geojson["features"]:
                    name = _lad_name(feat["properties"])
                    name = LAD_PARENT.get(name, name)
                    if name not in visible_areas:
                        continue
                    a = agg_lookup.loc[name]
                    feat["properties"]["area"] = name
                    feat["properties"]["mean_risk"] = round(float(a["mean_risk"]), 1)
                    feat["properties"]["n_lsoas"] = int(a["n_lsoas"])
                    feat["properties"]["n_top"] = int(a["n_top"])
                    rendered.append(feat)
                    if show_top_hotspots and name in top10_lads:
                        halo.append(feat)
                rendered_geo = {"type": "FeatureCollection", "features": rendered}

                if halo:
                    folium.GeoJson(
                        {"type": "FeatureCollection", "features": halo},
                        name="Top LADs halo",
                        style_function=lambda f: {
                            "fillOpacity": 0, "color": "#1F2937",
                            "weight": 2.5, "opacity": 0.8},
                        interactive=False).add_to(m)

                def style_fn_lad(feat):
                    # Greater London is our detailed-model region: paint it in a
                    # distinct blue with a bold outline so it reads as "London vs
                    # the rest of England", not as one more point on the red ramp.
                    if feat["properties"].get("area") == GREATER_LONDON:
                        return {"fillColor": LONDON_REGION_FILL,
                                "color": LONDON_REGION_LINE,
                                "weight": 2.2, "fillOpacity": 0.80}
                    val = feat["properties"].get("mean_risk", vmin)
                    val = min(max(val, vmin), vmax)  # clamp to ramp bounds
                    # light-grey stroke so the palest fills keep a visible edge
                    # against the white basemap instead of reading as "no data"
                    return {"fillColor": cmap(val), "color": "#9AA0A6",
                            "weight": 0.5, "fillOpacity": 0.85}

                folium.GeoJson(
                    rendered_geo, name="Local authorities",
                    style_function=style_fn_lad,
                    tooltip=folium.GeoJsonTooltip(
                        fields=["area", "mean_risk", "n_lsoas", "n_top"],
                        aliases=["Local authority", "Mean score",
                                 "LSOAs", "Top-tier LSOAs"],
                        localize=True, labels=True),
                    highlight_function=lambda f: {"weight": 2, "color": "#1F2937"},
                ).add_to(m)

                map_event = st_folium(m, width=None, height=600,
                                      returned_objects=["last_active_drawing"],
                                      key="lad_map")
                if map_event and map_event.get("last_active_drawing"):
                    drawing = map_event["last_active_drawing"]
                    clicked = drawing.get("properties", {}).get("area") \
                        or _lad_name(drawing.get("properties", {}))
                    click_sig = json.dumps(drawing, sort_keys=True, default=str)
                    if clicked and click_sig != st.session_state.last_consumed_lad_sig:
                        st.session_state.last_consumed_lad_sig = click_sig
                        st.session_state.selected_lad = clicked

        _detail_panel_lad(col_detail, df, lad_agg, GEO)


# ===== TIERS & GAPS TAB =====

with tab_tiers:
    st.markdown("### How the tiers break down")
    chart_col, gap_col = st.columns([2, 1], gap="medium")

    with chart_col:
        tier_counts = (
            df.groupby("tier")
            .agg(Neighbourhoods=("lsoa", "size"),
                 mean_risk=("display_score", "mean"))
            .reset_index())
        tier_counts["Tier"] = tier_counts["tier"].map(lambda t: f"{t}. {TIER_LABEL[t]}")
        chart = (
            alt.Chart(tier_counts).mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("Tier:N", sort=list(tier_counts["Tier"]), title=None),
                y=alt.Y("Neighbourhoods:Q", title="Neighbourhoods in view"),
                color=alt.Color("tier:N",
                    scale=alt.Scale(domain=list(TIER_COLOR.keys()),
                                    range=list(TIER_COLOR.values())),
                    legend=None),
                tooltip=["Tier", "Neighbourhoods",
                         alt.Tooltip("mean_risk:Q", title="Avg score", format=".1f")])
            .properties(height=240))
        st.altair_chart(chart, width="stretch")

    with gap_col:
        if GEO["has_stations"]:
            st.markdown("**Coverage gaps**")
            st.markdown(
                f"<span class='muted'>The 5 top-tier neighbourhoods furthest "
                f"from a police station (over {COVERAGE_THRESHOLD_KM} km).</span>",
                unsafe_allow_html=True)
            gaps = (
                df[df["tier"].isin({1}) & (df["dist_km"] > COVERAGE_THRESHOLD_KM)]
                .nlargest(5, "dist_km")
                [["lsoa_name", "area", "tier_label", "dist_km"]]
                .rename(columns={"dist_km": "km", "area": GEO["area_word"]}))
            if len(gaps):
                st.dataframe(gaps.assign(km=gaps["km"].round(1)),
                             hide_index=True, width="stretch")
            else:
                st.success(f"No top-tier neighbourhood is more than "
                           f"{COVERAGE_THRESHOLD_KM} km from a station in this view.")
        else:
            st.markdown("**Highest-demand local authorities**")
            st.markdown(
                "<span class='muted'>By share of LSOAs in the top tier.</span>",
                unsafe_allow_html=True)
            lad_rank = (
                df.groupby("area")
                .agg(top=("tier", lambda s: int((s == 1).sum())),
                     n=("lsoa", "size"),
                     mean_risk=("display_score", "mean"))
                .reset_index())
            lad_rank["top %"] = (100 * lad_rank["top"] / lad_rank["n"]).round(1)
            lad_rank = lad_rank.nlargest(8, "top %")[
                ["area", "top %", "n", "mean_risk"]].rename(
                columns={"area": "Local authority", "n": "LSOAs",
                         "mean_risk": "Mean"})
            st.dataframe(lad_rank.assign(Mean=lad_rank["Mean"].round(1)),
                         hide_index=True, width="stretch")


# ===== DYNAMIC DAY TAB (London only) =====

if tab_day is not None:
    with tab_day:
        st.markdown("### Conditions and dynamic factors")
        st.markdown(
            "<p class='muted'>Live London conditions and the date-specific "
            "factor layer. Each factor multiplies an LSOA's static demand for "
            "the chosen day; the map and officer split already reflect it. The "
            "validated index itself is never changed.</p>",
            unsafe_allow_html=True)

        st.markdown("#### Live conditions (now)")
        if weather is not None:
            w1, w2, w3, w4, w5 = st.columns(5)
            cc = weather.get("cloud_cover_pct")
            temp = weather.get("temp_c")
            precip = weather.get("precipitation_mm")
            wind = weather.get("wind_kmh")
            is_day = weather.get("is_day")
            sunrise = (weather.get("sunrise") or "")[-5:]
            sunset = (weather.get("sunset") or "")[-5:]
            w1.metric("Cloud cover", f"{cc:.0f}%" if cc is not None else "n/a",
                      delta=_sky_label(cc), delta_color="off",
                      help="Current London cloud cover from Open-Meteo, refreshed hourly.")
            w2.metric("Temperature", f"{temp:.1f}°C" if temp is not None else "n/a",
                      delta="warmer days lift ASB and theft" if (temp or 0) >= 15
                            else "cool day, lower ASB pressure", delta_color="off")
            w3.metric("Precipitation (1h)", f"{precip:.1f} mm" if precip is not None else "n/a",
                      delta="wet, suppresses outdoor crime" if (precip or 0) >= 0.2
                            else "dry", delta_color="off")
            w4.metric("Wind", f"{wind:.0f} km/h" if wind is not None else "n/a")
            w5.metric("Daylight", "Day" if is_day else "Night",
                      delta=(f"sunset {sunset}" if is_day else f"sunrise {sunrise}"),
                      delta_color="off",
                      help="Crime timing shifts between day and night "
                           "across crime types; daylight is a strong predictor of "
                           "burglary timing in particular.")
        else:
            st.caption("Weather widget offline (Open-Meteo unreachable). "
                       "Risk scoring is unaffected.")

        st.markdown("#### Active factors")
        if overlay_on and dyn_factor_meta:
            _fac_rows = [{
                "Factor": f["label"],
                "Multiplier": (f"×{f['multiplier']:.2f}" if f.get("multiplier")
                               else "per-area"),
                "Applies to": f["scope"],
                "Confidence": f["confidence"],
                "Status": "on" if f["active"] else "off",
            } for f in dyn_factor_meta]
            st.dataframe(pd.DataFrame(_fac_rows), hide_index=True,
                         width="stretch")
            st.caption(
                "risk_now = static index × product of the active factors "
                "(capped 0.5-2.5×). Tune each factor's strength in the sidebar "
                "→ Factor strengths. Weekend and football magnitudes are "
                "calibrated from MPS stop-and-search; the others are "
                "literature-informed planning assumptions (police data is "
                "monthly, so sub-monthly effects can't be measured locally). "
                + (f"Temperature is the daily mean for "
                   f"{overlay_date:%a %d %b %Y} (Open-Meteo "
                   f"{'archive' if pd.to_datetime(str(overlay_date)).date() < pd.Timestamp.now().date() else 'forecast'})."
                   if dyn_weather_dated else
                   "Temperature falls back to current live London "
                   "weather (no dated record for this day)."))
        elif overlay_on:
            st.caption("No factors are active for this date.")
        else:
            st.caption("Dynamic layer is off. Enable it in the sidebar.")

        st.markdown("#### What's on")
        if overlay_on and _otabs is not None and overlay_date is not None:
            _iso = pd.to_datetime(str(overlay_date)).date().isoformat()
            _m = _otabs["fixtures"]
            _m = _m[_m["date"] == _iso]
            _enames = sorted(_otabs["events"].loc[_otabs["events"]["date"] == _iso,
                                                  "event"].unique())
            _bits = [f"{r['club']} v {r['opponent']} ({r['kickoff']} KO"
                     f"{', derby' if r['is_derby'] else ''})" for _, r in _m.iterrows()]
            _bits += list(_enames)
            if pd.to_datetime(str(overlay_date)).weekday() in (4, 5):
                _bits.append("Friday/Saturday nightlife")
            _nboost = int((df["dyn_uplift"] >= 1).sum())
            if _bits and _nboost:
                # Show the realised rise (soft-capped), not the raw uplift, so
                # the "+N, now M" reads consistently with the capped score.
                df["dyn_shown"] = df["display_score"] - df.get(
                    "base_pre_overlay", df["display_score"])
                _top = df.loc[df["dyn_shown"].idxmax()]
                st.info(
                    f"**{overlay_date:%A %d %b %Y}**: " + ", ".join(_bits) + ".  \n"
                    f"{_nboost} neighbourhoods get a temporary lift. The biggest is "
                    f"{_top['lsoa_name']} (+{_top['dyn_shown']:.0f}, now "
                    f"{_top['display_score']:.0f}, {_top['tier_label']}). The map and "
                    "the officer split already account for it.")
                boosted = df[df["dyn_uplift"] >= 1].nlargest(12, "dyn_shown").copy()
                boosted["dyn_shown"] = boosted["dyn_shown"].round(0)
                boosted["display_score"] = boosted["display_score"].round(1)
                st.dataframe(
                    boosted[["lsoa_name", "area", "dyn_shown",
                             "display_score", "tier_label", "dyn_reasons"]]
                    .rename(columns={"lsoa_name": "Neighbourhood", "area": "Borough",
                                     "dyn_shown": "Uplift", "display_score": "Score now",
                                     "tier_label": "Tier", "dyn_reasons": "Why"}),
                    hide_index=True, width="stretch")
            else:
                st.caption(f"No major fixtures or events on {overlay_date:%d %b %Y}. "
                           "Pick a date in the 2025-26 season (try a Saturday).")
        else:
            st.caption("Event overlay is off. Turn it on in the sidebar to plan "
                       "for a specific day.")

        st.caption(
            "Match-day and Fri/Sat percentages are calibrated from MPS "
            "stop-and-search (pooled match day +22%, p≈0.05; weekend +12%, "
            "p<0.001). Heat and event sizes are literature and planning "
            "assumptions.")


# ===== METHOD TAB =====

with tab_method:
    st.markdown("### How this dashboard works")
    st.markdown(
        f"<p class='muted'>The whole pipeline at a glance. Every "
        f"neighbourhood (LSOA) in {GEO['scope']} gets a 0-100 risk score and "
        f"sits in a priority tier."
        + (" In London the score also drives an officer-allocation proposal "
           "and an optional day-by-day overlay." if GEO["has_allocation"]
           else "")
        + "</p>",
        unsafe_allow_html=True)

    # ----- pipeline flow diagram -----
    # Styled spans so the stages read in one horizontal sweep. Core stages use
    # the ink colour; the London-only stages (allocation, dynamic) use the
    # accent so they stand out as the extra layers that switch off for England.
    _ACCENT, _INK, _MUTED, _LINE = "#2563EB", "#1F2937", "#6B7280", "#E5E7EB"

    def _stage(label, sub, accent=False):
        edge = _ACCENT if accent else _LINE
        head = _ACCENT if accent else _INK
        return (
            f"<div style='flex:0 0 auto;min-width:96px;max-width:150px;"
            f"border:1px solid {edge};border-radius:9px;background:#FFFFFF;"
            f"box-shadow:0 1px 2px rgba(16,24,40,0.04);padding:0.5rem 0.7rem;"
            f"text-align:center;'>"
            f"<div style='font-size:0.82rem;font-weight:650;color:{head};"
            f"line-height:1.2;'>{label}</div>"
            f"<div style='font-size:0.7rem;color:{_MUTED};margin-top:0.15rem;"
            f"line-height:1.2;'>{sub}</div></div>")

    _arrow = (f"<div style='flex:0 0 auto;color:{_LINE};font-size:1.1rem;"
              f"align-self:center;padding:0 0.05rem;'>&#9654;</div>")

    _stages = [
        _stage("Data", "36 months"),
        _stage("Features", f"{len(FEATURES)} validated"),
        _stage("Weighted index", "0-100 score"),
        _stage("Supervised tiers", f"{len(TIER_LABEL)} priority bands"),
    ]
    if GEO["has_allocation"]:
        _stages.append(_stage("Allocation", "officers per BCU", True))
    if GEO["has_overlay"]:
        _stages.append(_stage("Dynamic", "per-date overlay", True))

    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;gap:0.35rem;align-items:stretch;"
        "margin:0.4rem 0 0.2rem;'>"
        + _arrow.join(_stages)
        + "</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<p class='muted' style='margin-top:0.3rem'>"
        + ("The two blue stages, allocation and the dynamic overlay, are "
           "London-only and switch off for England."
           if GEO["has_allocation"]
           else "Officer allocation and the dynamic overlay are London-only, "
                "so they are not part of this national view.")
        + "</p>",
        unsafe_allow_html=True)

    # ----- headline numbers -----
    _mk = [
        ("Neighbourhoods scored", f"{GEO['n_areas']:,}",
         {"help": "LSOAs given a 0-100 risk score across the whole scope."}),
        ("Model features", f"{len(FEATURES)}",
         {"help": "Validated drivers behind the weighted index."}),
        ("Priority tiers", f"{len(TIER_LABEL)}",
         {"help": "Supervised bands, from Highest to Low."}),
        ("Validation AUC", GEO["auc"],
         {"help": "How well the tier cut-points separate the LSOAs that went "
                  "on to become top-15% demand areas the following year."}),
    ]
    for _col, (_lab, _val, _kw) in zip(st.columns(len(_mk)), _mk):
        with _col:
            with st.container(border=True):
                st.metric(_lab, _val, **_kw)

    st.markdown("")

    # ----- concept cards -----
    _c1, _c2 = st.columns(2)
    with _c1:
        with st.container(border=True):
            st.markdown(f"**The {len(FEATURES)} model drivers**")
            st.markdown(
                "".join(
                    f"<div class='detail-row'>"
                    f"<span class='detail-label'>{label}</span>"
                    f"<span class='detail-value'>{GEO['weights'].get(key, '')}"
                    f"</span></div>"
                    for key, label in FEATURES.items()),
                unsafe_allow_html=True)
            st.caption(
                "Weights are the average of Negative Binomial regression "
                "coefficients and Random Forest importances. Features survived "
                "a Spearman correlation screen and a VIF multicollinearity "
                "check. Stop-and-search is kept out of the index on ethical "
                "grounds, so it cannot compound enforcement bias. Seasonality "
                "stays as diagnostic context and feeds the dynamic layer rather "
                "than the score.")
            if GEO["key"] == "england":
                st.caption(
                    "Crime volume is deliberately excluded, so the England "
                    "index reads as structural pressure rather than recorded "
                    "crime.")
    with _c2:
        with st.container(border=True):
            st.markdown(f"**The {len(TIER_LABEL)} priority tiers**")
            st.markdown(
                "<div class='legend' style='flex-direction:column;gap:0.4rem'>"
                + "".join(
                    f"<span><span class='sw' style='background:{TIER_COLOR[t]}'>"
                    f"</span>{t}. {TIER_LABEL[t]} "
                    f"<span class='muted'>(score {tier_range_txt(t)})</span>"
                    f"</span>"
                    for t in TIER_LABEL)
                + "</div>", unsafe_allow_html=True)
            st.caption(
                "Bands are contiguous, so a neighbourhood's score alone tells "
                "you its tier. The displayed index and the feature weights both "
                "use all 36 months (Apr 2023 to Mar 2026). Only the tier "
                "cut-points are supervised on a temporal split: the index is "
                "recomputed on the first 24 months, then the cut-points are "
                "chosen (ROC / Youden) against the LSOAs that became top-15% "
                f"demand areas over the next 12 months (AUC {GEO['auc']}).")
            st.markdown(
                f"<p class='muted'>Tier 1 holds {GEO['tier1_pct']} of LSOAs and "
                f"captured {GEO['tier1_recall']} of the next year's hotspots. "
                f"Read it as an allocation envelope, not a patrol route, and "
                f"rank inside it with the hotspot leaderboard.</p>",
                unsafe_allow_html=True)

    if GEO["has_allocation"]:
        with st.container(border=True):
            st.markdown("**Officer allocation**")
            _a1, _a2 = st.columns([3, 2])
            with _a1:
                st.markdown(
                    "Each neighbourhood is tied to the Metropolitan Police "
                    "Basic Command Unit (BCU) covering its borough. That BCU's "
                    "real frontline strength is shared across the BCU's LSOAs "
                    "in proportion to demand, then measured against an even "
                    "split of the same BCU's strength. A plus or minus figure "
                    "is officers above or below an equal share within that "
                    "command unit, never a move between BCUs.")
                st.caption(
                    "Whether a BCU is over- or under-resourced against "
                    "London-wide demand is a separate calculation, the BCU "
                    "allocative-efficiency analysis. The City of London is "
                    "policed by its own force, not a Met BCU, so its LSOAs are "
                    "scored for demand but carry no allocation proposal.")
            with _a2:
                st.markdown(
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Strength source</span>"
                    "<span class='detail-value'>Officer + PCSO FTE</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Reference</span>"
                    "<span class='detail-value'>MPS report, 31 Mar 2026</span>"
                    "</div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Compared against</span>"
                    "<span class='detail-value'>Even split within BCU</span>"
                    "</div>",
                    unsafe_allow_html=True)

    if GEO["map_mode"] == "lad":
        with st.container(border=True):
            st.markdown("**National map and Greater London**")
            st.info(
                "England is mapped at local-authority level. 33,755 LSOA "
                "polygons cannot render in the browser, so each authority takes "
                "the mean risk score of its LSOAs. The leaderboard and detail "
                "panel stay at full LSOA resolution. There is no "
                "officer-allocation layer outside London, since police strength "
                "is wired in only at Met BCU level, so the national view shows "
                "demand and tiers only.")
            st.markdown(
                "On the national map, London's 33 local authorities are "
                "dissolved into one Greater London region, drawn in a distinct "
                "blue instead of on the red ramp. It is scored with the same "
                "England-wide structural model as everywhere else, so it can be "
                "compared with the rest of England on equal terms. The full "
                "four-feature London index, the per-LSOA tiers and the BCU "
                "officer allocation all live under the London scope.")

    if GEO["has_overlay"]:
        with st.container(border=True):
            st.markdown("**Dynamic factor layer**")
            _d1, _d2 = st.columns([3, 2])
            with _d1:
                st.markdown(
                    "The static index reflects 36 months of demand. On top of "
                    "it, an optional layer multiplies each LSOA's score for a "
                    "chosen date:")
                st.markdown(
                    "`risk_now = static × clamp(product of factors, 0.5, 2.5)`")
                st.caption(
                    "Weekend and match-day strengths are calibrated from MPS "
                    "stop-and-search. The rest are literature-informed planning "
                    "assumptions, each an officer-tunable multiplier from a "
                    "shared config (dynamic_factor_config.json). It never "
                    "changes the validated model. The boosted score lands on a "
                    "saturating 0-100 scale, so it approaches 100 as the uplift "
                    "grows rather than piling up at a flat ceiling, and the "
                    "busiest neighbourhoods stay apart on a heavily stacked "
                    "day.")
            with _d2:
                st.markdown(
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Temperature</span>"
                    "<span class='detail-value'>multiplier</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Weekend</span>"
                    "<span class='detail-value'>multiplier</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Fri/Sat nightlife</span>"
                    "<span class='detail-value'>absorbs weekend</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Premier League match</span>"
                    "<span class='detail-value'>multiplier</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Major events</span>"
                    "<span class='detail-value'>multiplier</span></div>"
                    "<div class='detail-row'>"
                    "<span class='detail-label'>Bank holiday</span>"
                    "<span class='detail-value'>multiplier</span></div>",
                    unsafe_allow_html=True)
            st.caption(
                "In nightlife LSOAs the nightlife factor absorbs the weekend "
                "factor instead of stacking, so each area gets one Friday-night "
                "uplift, not two.")


# ----- footer -----

st.markdown("---")
_src = ("data.police.uk crime and outcomes, NASA VIIRS Black Marble "
        "night-time lights, IMD 2025, Met stop-and-search (dynamic-layer "
        "calibration only) and Open-Meteo weather"
        if GEO["key"] == "london" else
        "data.police.uk crime and outcomes (England), NASA VIIRS Black "
        "Marble night-time lights and IMD 2025")
st.caption(
    f"Built on {_src}. "
    f"{n_total:,} of {len(phase5):,} neighbourhoods in view. Proof of concept, CBL Group 16.")
