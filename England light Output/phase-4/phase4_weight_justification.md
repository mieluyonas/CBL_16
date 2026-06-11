# Phase-4: Weight Justification (England_final_light)

## Model Description
This is a **structural demand index** — it captures socio-economic drivers of policing demand
without using crime volume (severity_weighted_count). This allows comparison of areas by their
underlying structural pressures independently of observed crime, providing a complementary
perspective to the volume-weighted England model.

## Method
Final weights are the average of two independently derived weight sets:
1. **Negative Binomial Regression** (primary) — standardised coefficients from a NB regression of crime_count on validated features.
2. **Random Forest importances** (cross-validation) — feature importances from a 300-tree RF regressor.
   RF 5-fold CV R2: 0.2724 +/- 0.1141
   NB vs RF rank correlation: r=1.0000

## Final Weights

| Feature | NB weight | RF weight | Final weight | NB p-value | Significant |
|---|---|---|---|---|---|
| ntl_mean_radiance | 0.4235 | 0.4804 | 0.4519 | 0.0000 | Yes |
| resolution_rate | 0.3083 | 0.2663 | 0.2873 | 0.0000 | Yes |
| employment_deprivation | 0.2683 | 0.2534 | 0.2608 | 0.0000 | Yes |

*Features flagged as non-significant are retained because they passed Phase 3 Spearman validation.

## Feature Direction
- All features normalised to [0,1] min-max before weighting.
- `employment_rank` was INVERTED (employment_deprivation = max_rank + 1 - employment_rank) so higher value = more deprived = higher demand.
- `resolution_rate` is positively associated with demand.

## Features Dropped by Phase 3 VIF
- `employment_rank` — VIF exceeded threshold; dropped before Phase 4.
- `income_rank` — VIF exceeded threshold; dropped before Phase 4.

## Features Excluded from this Model by Design
- `severity_weighted_count` — excluded by design: this index measures structural demand, not crime volume.
- `stop_search_rate` — excluded on ethical grounds (potential for compounding racial bias).
- `total_footfall` — excluded as TfL data is London-specific; no national equivalent available.
- `seasonal_volatility` — excluded by group decision.

## Sensitivity Analysis
- Mean top-10% rank stability across all +-10% perturbations: **97.8%**
- Mean bottom-10% rank stability: **98.1%**

## Literature Cross-Reference
- Laufs et al. (2021) identify deprivation as a primary structural driver of police demand — consistent with the weight on employment_deprivation.
- Resolution rate reflects systemic capacity and is included as a demand-side pressure indicator.