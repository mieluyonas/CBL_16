"""
12_event_overlay.py: dynamic event overlay for the London demand dashboard.

Builds a date-keyed, per-LSOA demand *uplift* layer that sits ON TOP of the
team's static Phase 5 risk index. This is deliberately NOT a new static
feature: the validated index (Phase 3 Spearman/VIF, Phase 4 NB+RF weights) is
left untouched. The overlay answers the mid-term feedback "make patrol
dynamic". For a chosen date it temporarily raises demand in LSOAs that are
hosting:

  * a Premier League home match (7 London clubs),
  * a major recurring event (Notting Hill Carnival, NYE, Marathon, Pride),
  * Friday/Saturday nightlife (scaled by pub/club density per LSOA),
  * a hot day (heat-aggression uplift on violence-prone LSOAs).

HONESTY / DATA LIMIT
--------------------
data.police.uk is published at MONTHLY granularity (by design, for victim
anonymisation). We therefore *cannot* measure a match-day or Friday-night
spike from our own data. Every multiplier below is a literature-informed
PLANNING ASSUMPTION, exposed as a tunable parameter, not a fitted value.
The references that justify the *direction* of each effect live in
EVENT_OVERLAY.md; the magnitudes still need a sourced citation before any
number is quoted as fact.

Outputs (analysis/out/):
  event_venues.parquet     stadium -> affected LSOA(s) with spillover weight
  london_fixtures.parquet  2025-26 PL home fixtures for the 7 London clubs
  london_events.parquet    curated major recurring events -> affected LSOAs
  lsoa_nightlife.parquet    pub/bar/nightclub count per LSOA (night modulator)

Run once to build everything:  uv run python 12_event_overlay.py
The dashboard imports load_overlay_tables() and compute_overlay_config() here.
"""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: geopandas / shapely / requests are imported lazily inside the build
# functions only. The dashboard imports this module at runtime for
# load_overlay_tables() + compute_overlay_config(), pure pandas/numpy, so
# the Space never loads geopandas (keeps it under the 1 GB free-tier memory).

ROOT = Path(__file__).parent
ASSETS = ROOT / "dashboard_assets"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

LSOA_GEOJSON = ASSETS / "london_lsoa.geojson"
FIXTURES_URL = "https://fixturedownload.com/download/epl-2025-GMTStandardTime.csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ---------------------------------------------------------------------------
# Uplifts are expressed as a PERCENTAGE OF EACH LSOA'S BASELINE demand score,
# so every number is interpretable ("a match day adds ~22% to this area's
# demand for the day") and traceable to a source. Source tag per component:
#   DATA   = estimated from MPS stop-and-search (13_overlay_validation.py)
#   LIT    = literature-informed multiplier
#   ASSUME = planning assumption, no external source confirmed yet
# load_calibration() overwrites the DATA rows with the empirical values.
# ---------------------------------------------------------------------------
PARAMS = {
    "football_pct": 22.0,            # DATA: pooled match-day S&S uplift
    "football_evening_mult": 1.15,   # LIT: evening kick-offs (alcohol, low light)
    "football_derby_mult": 1.15,     # LIT: London derbies
    "football_spillover": 0.45,      # geometric: LSOAs near (not at) the ground
    "spillover_radius_m": 1100,
    "event_pct": 45.0,               # ASSUME: scaled by per-event magnitude
    "event_spillover": 0.50,
    "event_radius_m": 1500,
    "weekend_pct": 12.0,             # DATA: Fri/Sat S&S uplift, distributed by nightlife
    "heat_pct": 15.0,                # LIT: Rotton & Cohn, scaled by the heat factor
    "heat_onset_c": 22.0,            # only genuinely warm London days trigger it
    "heat_peak_c": 28.0,
}
PARAM_SOURCE = {"football_pct": "MPS S&S", "weekend_pct": "MPS S&S",
                "heat_pct": "literature", "event_pct": "planning assumption"}


def load_calibration(params: dict = PARAMS) -> dict:
    """Overwrite the DATA-tagged percentages with the empirical estimates from
    13_overlay_validation.py, if its calibration file has been built."""
    f = OUT / "overlay_calibration.json"
    if f.exists():
        c = json.loads(f.read_text())
        if c.get("football_pct_pooled"):
            params["football_pct"] = float(c["football_pct_pooled"])
        if c.get("weekend_pct"):
            params["weekend_pct"] = float(c["weekend_pct"])
    return params


load_calibration()  # apply empirical calibration at import if available

# 7 Premier League London clubs. Capacity drives the relative football uplift.
STADIUMS = {
    "Arsenal":        ("Emirates Stadium",            51.5549, -0.1084, 60704),
    "Chelsea":        ("Stamford Bridge",             51.4817, -0.1910, 40343),
    "Tottenham":      ("Tottenham Hotspur Stadium",   51.6043, -0.0665, 62850),
    "West Ham":       ("London Stadium",              51.5387, -0.0166, 62500),
    "Crystal Palace": ("Selhurst Park",               51.3983, -0.0855, 25486),
    "Fulham":         ("Craven Cottage",              51.4749, -0.2216, 29600),
    "Brentford":      ("Gtech Community Stadium",      51.4906, -0.2889, 17250),
}
_MAX_CAP = max(c for *_, c in STADIUMS.values())

# Curated major recurring London events. Dates are the 2025-26 cycle; the
# rule (e.g. "August bank-holiday weekend") is what recurs, so verify per year.
# magnitude is a 0-1 planning weight; demand_type flags crime vs crowd/medical.
EVENTS = [
    # name, date(s) ISO, lat, lon, magnitude, demand_type
    ("Notting Hill Carnival", ["2025-08-24", "2025-08-25"], 51.5200, -0.2050, 1.00, "public-order"),
    ("New Year's Eve fireworks", ["2025-12-31"],            51.5033, -0.1196, 0.95, "public-order"),
    ("Pride in London",        ["2025-07-05"],              51.5103, -0.1340, 0.65, "public-order"),
    ("London Marathon",        ["2026-04-26"],              51.5014, -0.1419, 0.55, "crowd/medical"),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _load_lsoa_gdf():
    import geopandas as gpd
    gj = json.load(open(LSOA_GEOJSON))
    gdf = gpd.GeoDataFrame.from_features(gj["features"]).set_crs("EPSG:4326")
    return gdf[["LSOA21CD", "LSOA21NM", "geometry"]]


def _point_to_lsoa(gdf_m, lat: float, lon: float) -> str:
    """LSOA21CD containing (lat, lon); nearest polygon if the point falls in a
    simplification gap (e.g. riverside grounds like Craven Cottage)."""
    import geopandas as gpd
    from shapely.geometry import Point
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(27700).iloc[0]
    hit = gdf_m[gdf_m.contains(pt)]
    if len(hit):
        return hit.iloc[0]["LSOA21CD"]
    dists = gdf_m.geometry.distance(pt)
    return gdf_m.loc[dists.idxmin(), "LSOA21CD"]


def _nearby_lsoas(gdf_m, lat: float, lon: float, radius_m: float) -> list[str]:
    """LSOA codes whose centroid is within radius_m of (lat, lon)."""
    import geopandas as gpd
    from shapely.geometry import Point
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(27700).iloc[0]
    d = gdf_m.geometry.centroid.distance(pt)
    return gdf_m.loc[d <= radius_m, "LSOA21CD"].tolist()


# ---------------------------------------------------------------------------
# Builders (each caches to out/; skip if present unless rebuild=True)
# ---------------------------------------------------------------------------
def build_venues(gdf_m: gpd.GeoDataFrame, rebuild: bool = False) -> pd.DataFrame:
    out = OUT / "event_venues.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    rows = []
    for club, (name, lat, lon, cap) in STADIUMS.items():
        home_lsoa = _point_to_lsoa(gdf_m, lat, lon)
        near = set(_nearby_lsoas(gdf_m, lat, lon, PARAMS["spillover_radius_m"]))
        near.add(home_lsoa)
        for code in near:
            rows.append({
                "club": club, "stadium": name, "capacity": cap,
                "lsoa21cd": code,
                "weight": 1.0 if code == home_lsoa else PARAMS["football_spillover"],
                "is_ground": code == home_lsoa,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[venues] {df['club'].nunique()} clubs -> {len(df)} LSOA rows -> {out.name}")
    return df


def build_fixtures(rebuild: bool = False) -> pd.DataFrame:
    out = OUT / "london_fixtures.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    # fixturedownload blocks urllib's default UA, so fetch via requests.
    import io
    import requests
    resp = requests.get(FIXTURES_URL, timeout=30,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    # fixturedownload uses some short names; map them to our stadium keys.
    aliases = {"Spurs": "Tottenham"}
    raw["Home Team"] = raw["Home Team"].replace(aliases)
    raw["Away Team"] = raw["Away Team"].replace(aliases)
    raw["dt"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y %H:%M")
    london = raw[raw["Home Team"].isin(STADIUMS)].copy()
    london["date"] = london["dt"].dt.date.astype(str)
    london["kickoff"] = london["dt"].dt.strftime("%H:%M")
    london["is_evening"] = london["dt"].dt.hour >= 17
    london["is_derby"] = london["Away Team"].isin(STADIUMS)
    london = london.rename(columns={"Home Team": "club", "Away Team": "opponent"})
    df = london[["date", "kickoff", "club", "opponent",
                 "is_evening", "is_derby"]].reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"[fixtures] {len(df)} London home matches "
          f"({df['date'].min()} … {df['date'].max()}) -> {out.name}")
    return df


def build_events(gdf_m: gpd.GeoDataFrame, rebuild: bool = False) -> pd.DataFrame:
    out = OUT / "london_events.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    rows = []
    for name, dates, lat, lon, mag, dtype in EVENTS:
        home = _point_to_lsoa(gdf_m, lat, lon)
        near = set(_nearby_lsoas(gdf_m, lat, lon, PARAMS["event_radius_m"]))
        near.add(home)
        for d in dates:
            for code in near:
                rows.append({
                    "event": name, "date": d, "magnitude": mag,
                    "demand_type": dtype, "lsoa21cd": code,
                    "weight": 1.0 if code == home else PARAMS["event_spillover"],
                })
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[events] {df['event'].nunique()} events -> {len(df)} LSOA-date rows -> {out.name}")
    return df


def build_nightlife(gdf_m: gpd.GeoDataFrame, rebuild: bool = False) -> pd.DataFrame:
    out = OUT / "lsoa_nightlife.parquet"
    if out.exists() and not rebuild:
        return pd.read_parquet(out)
    query = """
    [out:json][timeout:120];
    (
      node["amenity"~"pub|bar|nightclub"](51.28,-0.51,51.69,0.34);
      way["amenity"~"pub|bar|nightclub"](51.28,-0.51,51.69,0.34);
    );
    out center;
    """
    import geopandas as gpd
    import requests
    print("[nightlife] querying Overpass for pubs/bars/nightclubs…")
    r = requests.post(OVERPASS_URL, data={"data": query}, timeout=180,
                      headers={"User-Agent": "CBL16-dashboard/1.0 (TU/e student project)"})
    r.raise_for_status()
    elems = r.json().get("elements", [])
    pts = []
    for e in elems:
        lat = e.get("lat") or e.get("center", {}).get("lat")
        lon = e.get("lon") or e.get("center", {}).get("lon")
        if lat is not None and lon is not None:
            pts.append((lon, lat))
    print(f"[nightlife] {len(pts)} venues; assigning to LSOAs…")
    venues = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([p[0] for p in pts], [p[1] for p in pts]),
        crs="EPSG:4326").to_crs(27700)
    joined = gpd.sjoin(venues, gdf_m[["LSOA21CD", "geometry"]],
                       how="inner", predicate="within")
    counts = joined.groupby("LSOA21CD").size().rename("n_venues")
    df = gdf_m[["LSOA21CD"]].merge(counts, on="LSOA21CD", how="left").fillna({"n_venues": 0})
    # nightlife_score: percentile rank 0-1, so the night uplift lands on the
    # pubs/clubs LSOAs (Soho, Shoreditch, Clapham) rather than flat across London.
    # Zero-venue LSOAs must score exactly 0. pct-rank gives ties the AVERAGE
    # rank, which would hand the ~3,200 venue-less LSOAs a phantom ~0.32.
    df["nightlife_score"] = df["n_venues"].rank(pct=True)
    df.loc[df["n_venues"] == 0, "nightlife_score"] = 0.0
    df = df.rename(columns={"LSOA21CD": "lsoa21cd"})
    df.to_parquet(out, index=False)
    print(f"[nightlife] {int(df['n_venues'].sum())} venues over "
          f"{(df['n_venues'] > 0).sum()} LSOAs -> {out.name}")
    return df


# ---------------------------------------------------------------------------
# Overlay computation (imported by the dashboard)
# ---------------------------------------------------------------------------
def load_overlay_tables() -> dict[str, pd.DataFrame]:
    """Load the four prebuilt tables. Assumes 12_event_overlay.py has run."""
    return {
        "venues": pd.read_parquet(OUT / "event_venues.parquet"),
        "fixtures": pd.read_parquet(OUT / "london_fixtures.parquet"),
        "events": pd.read_parquet(OUT / "london_events.parquet"),
        "nightlife": pd.read_parquet(OUT / "lsoa_nightlife.parquet"),
    }


# ---------------------------------------------------------------------------
# Config-driven MULTIPLICATIVE layer. Wires Mateus's
# phase7/dynamic_factor_config.json. risk_now = risk_static × clamp(Π m, .5, 2.5)
# The officer "strength" slider per factor scales that factor's deviation from
# 1.0 (0 = off, 1 = the literature/data default, 2 = double). Returns the same
# (lsoa21cd, uplift, reasons) shape the dashboard expects, where
# uplift = baseline × (combined_factor − 1) so the additive merge in the
# dashboard reproduces the multiplicative product.
# ---------------------------------------------------------------------------
FACTOR_CONFIG_PATH = ROOT / "dynamic_factor_config.json"
BANK_HOLIDAYS_PATH = ROOT / "data_external" / "uk_bank_holidays.json"
_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SUNDAY_MULT = 1.05  # literature; Fri/Sat weekend + match-day come from the
                    # MPS stop-and-search calibration in PARAMS (load_calibration)
# Mirrors the config so the engine still runs if the JSON is absent.
_FALLBACK_BOUNDS = {
    "temperature": (0.85, 1.30),
    "weekend": (1.00, 1.35), "holiday": (0.90, 1.30),
    "nightlife_alcohol": (1.00, 1.50), "football": (1.00, 1.60),
    "events": (1.00, 1.60),
}


def load_factor_config(path: Path = FACTOR_CONFIG_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def load_bank_holidays(path: Path = BANK_HOLIDAYS_PATH) -> set[str]:
    """England & Wales bank-holiday ISO dates (gov.uk bank-holidays.json)."""
    try:
        d = json.loads(Path(path).read_text())
        return {e["date"] for e in d["england-and-wales"]["events"]}
    except Exception:
        return set()


def compute_overlay_config(target: str | _date, tables: dict[str, pd.DataFrame],
                           baseline: "pd.Series | None" = None, *,
                           temp_c: float | None = None,
                           cloud_cover_pct: float | None = None,
                           is_night: bool = True,
                           strengths: dict | None = None,
                           config: dict | None = None,
                           holidays: "set[str] | None" = None) -> pd.DataFrame:
    """Per-LSOA multiplicative dynamic layer for one date.

    combined_factor(lsoa) = clamp( Π active factor multipliers, cmin, cmax );
    uplift = baseline × (combined_factor − 1). Active factors and their global
    multipliers are attached as the returned frame's .attrs['factors'].
    """
    config = config if config is not None else load_factor_config()
    holidays = holidays if holidays is not None else load_bank_holidays()
    strengths = strengths or {}
    caps = (config or {}).get("global_caps", {})
    cmin, cmax = float(caps.get("combined_min", 0.5)), float(caps.get("combined_max", 2.5))
    cfg_bounds = {f["id"]: f for f in (config or {}).get("factors", [])}

    def lohi(fid: str) -> tuple[float, float]:
        f = cfg_bounds.get(fid)
        if f:
            return float(f.get("min_multiplier", _FALLBACK_BOUNDS[fid][0])), \
                   float(f.get("max_multiplier", _FALLBACK_BOUNDS[fid][1]))
        return _FALLBACK_BOUNDS[fid]

    def eff(base_mult: float, fid: str) -> float:
        s = float(strengths.get(fid, 1.0))
        lo, hi = lohi(fid)
        return min(max(1.0 + s * (base_mult - 1.0), lo), hi)

    d = pd.to_datetime(str(target)).date()
    iso, dow, dname = d.isoformat(), d.weekday(), _DOW[pd.to_datetime(str(target)).weekday()]
    DEFAULT_BASE = 50.0
    if baseline is None:
        baseline = pd.Series(DEFAULT_BASE,
                             index=pd.Index(tables["nightlife"]["lsoa21cd"].unique()))
    idx = baseline.index

    gmult = 1.0
    meta: list[dict] = []
    applied: dict[str, float] = {}  # effective multiplier per active global factor

    def add_global(fid, base_mult, label, conf):
        nonlocal gmult
        m = eff(base_mult, fid)
        active = abs(m - 1.0) > 0.005
        meta.append({"id": fid, "label": label, "multiplier": round(m, 3),
                     "scope": "all areas", "confidence": conf, "active": active})
        if active:
            gmult *= m
            applied[fid] = m

    if temp_c is not None:
        add_global("temperature", 1 + 0.012 * (temp_c - 15),
                   f"Temperature {temp_c:.0f}°C", "medium")
    if dname in ("Fri", "Sat"):
        add_global("weekend", 1 + PARAMS["weekend_pct"] / 100.0,
                   f"{dname} (weekend)", "data (MPS S&S)")
    elif dname == "Sun":
        add_global("weekend", SUNDAY_MULT, "Sun (weekend)", "literature")
    if iso in holidays:
        add_global("holiday", 1.10, "Bank holiday", "low-medium")

    local: dict[str, float] = {}
    local_why: dict[str, list[str]] = {}

    def add_local(code, m, label):
        if m <= 1.005:
            return
        local[code] = local.get(code, 1.0) * m
        local_why.setdefault(code, []).append(f"{label} ×{m:.2f}")

    if dow in (4, 5) and is_night:
        # Weekend and nightlife both tell the "it's Friday night" story, so
        # they must not stack (group decision, 10 Jun). The weekend factor is
        # the citywide Fri/Sat baseline; nightlife only adds the EXCESS over
        # that baseline in venue-dense LSOAs. Total there = max(weekend,
        # nightlife target), never weekend × nightlife.
        wk = applied.get("weekend", 1.0)
        nl = tables["nightlife"]
        # Guard against prebuilt parquets where pct-rank ties gave zero-venue
        # LSOAs a non-zero score (see build_nightlife): no venues, no uplift.
        if "n_venues" in nl.columns:
            nl = nl[nl["n_venues"] > 0]
        for _, n in nl[nl["nightlife_score"] > 0].iterrows():
            target = eff(1 + 0.40 * n["nightlife_score"], "nightlife_alcohol")
            add_local(n["lsoa21cd"], max(1.0, target / wk),
                      "Fri/Sat nightlife (over weekend base)")
        meta.append({"id": "nightlife_alcohol", "label": "Fri/Sat nightlife",
                     "multiplier": None,
                     "scope": "nightlife LSOAs (absorbs the weekend factor)",
                     "confidence": "medium-high",
                     "active": float(strengths.get("nightlife_alcohol", 1.0)) > 0})

    todays = tables["fixtures"][tables["fixtures"]["date"] == iso]
    if len(todays):
        ven = tables["venues"]
        for _, mt in todays.iterrows():
            # S&S-calibrated match-day effect (+football_pct of baseline),
            # with literature evening-KO and derby bumps on top.
            base = 1 + PARAMS["football_pct"] / 100.0
            extra = []
            if mt.get("is_evening"):
                base *= PARAMS["football_evening_mult"]; extra.append("evening")
            if mt.get("is_derby"):
                base *= PARAMS["football_derby_mult"]; extra.append("derby")
            sfx = (" " + "+".join(extra)) if extra else ""
            for _, v in ven[ven["club"] == mt["club"]].iterrows():
                add_local(v["lsoa21cd"], eff(1 + (base - 1) * v["weight"], "football"),
                          f"{mt['club']} v {mt['opponent']} (match{sfx})")
        meta.append({"id": "football", "label": "Football match day",
                     "multiplier": None, "scope": "near grounds",
                     "confidence": "data (MPS S&S) + lit",
                     "active": float(strengths.get("football", 1.0)) > 0})

    evs = tables["events"][tables["events"]["date"] == iso]
    if len(evs):
        for _, e in evs.iterrows():
            add_local(e["lsoa21cd"], eff(1 + 0.45 * e["magnitude"] * e["weight"], "events"),
                      str(e["event"]))
        meta.append({"id": "events", "label": "Major event",
                     "multiplier": None, "scope": "event LSOAs",
                     "confidence": "planning",
                     "active": float(strengths.get("events", 1.0)) > 0})

    combined = pd.Series(gmult, index=idx, dtype=float)
    if local:
        combined = combined * pd.Series(local).reindex(idx).fillna(1.0)
    combined = combined.clip(cmin, cmax)
    base = baseline.reindex(idx).astype(float).fillna(DEFAULT_BASE)
    uplift = base * (combined - 1.0)

    gtxt = "; ".join(f"{f['label']} ×{f['multiplier']:.2f}"
                     for f in meta if f.get("multiplier") and f["active"])
    reasons = ["; ".join(([gtxt] if gtxt else []) + local_why.get(c, [])) for c in idx]

    out = pd.DataFrame({"lsoa21cd": list(idx), "uplift": uplift.values,
                        "combined_factor": combined.values, "reasons": reasons})
    out = out[out["uplift"].abs() > 1e-9].reset_index(drop=True)
    out.attrs["factors"] = meta
    return out.sort_values("uplift", ascending=False).reset_index(drop=True)


def main() -> None:
    gdf = _load_lsoa_gdf().to_crs(27700)
    venues = build_venues(gdf)
    fixtures = build_fixtures()
    events = build_events(gdf)
    nightlife = build_nightlife(gdf)

    # Demo: a Saturday with a home match, to sanity-check the config layer.
    tables = {"venues": venues, "fixtures": fixtures,
              "events": events, "nightlife": nightlife}
    sample = fixtures.iloc[0]["date"] if len(fixtures) else "2025-08-23"
    demo = compute_overlay_config(sample, tables, temp_c=24.0,
                                  cloud_cover_pct=30.0, is_night=True)
    print(f"\n[demo] dynamic factors for {sample} (24°C):")
    for f in demo.attrs.get("factors", []):
        print("   ", f["id"], f.get("multiplier"), f["active"])
    print(demo.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
