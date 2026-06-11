# Phase-4: Weight Justification (London-final-light)

London-final-light is an ablation of the main London model that **excludes `stop_search_rate` and `seasonal_volatility`**.

## Method
Final weights are the average of two independently derived weight sets:
1. **Negative Binomial Regression** (primary) standardised coefficients from a NB regression of crime_count on validated features. NB is used because crime counts are overdispersed count data (Laufs et al., 2021).
2. **Random Forest importances** (cross-validation) feature importances from a 300-tree RF regressor trained on the same features.
   RF 5-fold CV R2: 0.7607 +/- 0.1316
   NB vs RF rank correlation: r=0.8000

## Final Weights

| Feature | NB weight | RF weight | Final weight | NB p-value | Significant |
|---|---|---|---|---|---|
| severity_weighted_count | 0.8114 | 0.7227 | 0.7670 | 0.0000 | Yes |
| ntl_mean_radiance | 0.0713 | 0.1891 | 0.1302 | 0.0000 | Yes |
| employment_deprivation | 0.1108 | 0.0443 | 0.0775 | 0.0000 | Yes |
| resolution_rate | 0.0066 | 0.0439 | 0.0253 | 0.0107 | Yes |

*Features flagged as non-significant are retained because they passed Phase 3 Spearman validation and are theoretically justified by the literature.

## Features Excluded from this Model
- `stop_search_rate`: excluded on ethical grounds (potential for compounding racial bias in demand estimates).
- `seasonal_volatility`: excluded by group decision (seasonality retained only as diagnostic context in Phase 3).

## Feature Direction
- All features normalised to [0,1] min-max before weighting.
- Any `*_rank` deprivation column was INVERTED (deprivation = max_rank + 1 - rank) so higher value = more deprived = higher demand.
- All other features are positively associated with policing demand.

## Sensitivity Analysis
- Mean top-10% rank stability across all +-10% perturbations: **98.5%**
- Mean bottom-10% rank stability: **98.9%**
- High stability confirms the index is robust to small changes in weights.

## Literature Cross-Reference
- Laufs et al. (2021) identify crime volume, severity, and deprivation as the primary drivers of police demand.
- Cambridge Crime Harm Index (Sherman et al., 2016) underpins the severity_weighted_count feature.