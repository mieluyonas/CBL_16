# CBL-16 demand dashboard

Streamlit dashboard for the final model: London + England demand index,
supervised priority tiers (Phase 6), BCU officer allocation and the
dynamic-day factor layer.

Live deployment: https://johnhopkinsun-cbl-16-dashboard.hf.space

## Run locally

```
pip install -r requirements.txt
streamlit run 10_dashboard.py
```

Everything the app needs is in this folder. The app does not need network
access at runtime, apart from the live weather strip (Open-Meteo, no key).

## What's here

| path | what it is |
|---|---|
| `10_dashboard.py` | the Streamlit app (deployed to HF as `src/streamlit_app.py`) |
| `12_event_overlay.py` | dynamic-day factor layer (weekend/nightlife/football/events/weather) |
| `team_model/`, `england_model/` | final model parquets: `tier` (supervised), features, scores |
| `out/` | overlay tables (fixtures, events, nightlife, venues), calibration, BCU FTE, per-crime-type pivot for the weight sliders |
| `dashboard_assets/` | LSOA/LAD boundaries (ONS), police stations (OSM), CSS |
| `data_external/uk_bank_holidays.json` | bank holidays for the dynamic layer |
| `dynamic_factor_config.json` | officer-tunable factor config (from phase 7) |

## Tier semantics (important)

- **Priority tiers** (the `tier` column, 1 Highest … 4 Low) are the Phase 6
  **supervised** tiers: boundaries calibrated on a 24-month fit against the
  LSOAs that became top-15% demand in the following 12 months (AUC 0.92
  London / 0.83 England; Tier 1 captured 81.7% / 77.2% of realised future
  hotspots). The filters, KPIs, allocation and the map all run off them.
  Tier 1 covers 25.3% of London LSOAs (34.8% England), so it works as
  an allocation envelope; patrol-level targeting goes through the top-N leaderboard.

The model parquets still carry a Phase 5 K-Means `profile` column, but the
dashboard no longer surfaces it; the K-Means typology stays in the report as
a descriptive analysis only.

Full design rationale, validation numbers and Q&A prep live in the
analysis repo (`DASHBOARD_GUIDE.md`, `TIERING_DECISION.md`).
