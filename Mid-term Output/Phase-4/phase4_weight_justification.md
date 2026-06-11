# Phase-4: Weight Justification

## Method
Final weights are the average of two independently derived weight sets:
1. **Negative Binomial Regression** (primary) standardised coefficients from a NB regression of crime_count on validated features. NB is used because crime counts are overdispersed count data (Laufs et al., 2021).
2. **Random Forest importances** (cross-validation) feature importances from a 300-tree RF regressor trained on the same features.
   RF 5-fold CV R2: 0.8655 +/- 0.1351
   NB vs RF rank correlation: r=0.8286

## Final Weights

| Feature | NB weight | RF weight | Final weight | NB p-value | Significant |
|---|---|---|---|---|---|
| severity_weighted_count | 0.8172 | 0.4200 | 0.6186 | 0.0000 | Yes |
| seasonal_volatility | 0.0769 | 0.3358 | 0.2063 | 0.0000 | Yes |
| total_footfall | 0.0136 | 0.1044 | 0.0590 | 0.0000 | Yes |
| stop_search_rate | 0.0135 | 0.0937 | 0.0536 | 0.0000 | Yes |
| employment_deprivation | 0.0759 | 0.0253 | 0.0506 | 0.0000 | Yes |
| resolution_rate | 0.0029 | 0.0209 | 0.0119 | 0.1858 | No* |

*Features flagged as non-significant are retained because they passed Phase 3 Spearman validation and are theoretically justified by the literature (Laufs et al., 2021; What Works Centre for Crime Reduction).

## Feature Direction
- All features normalised to [0,1] min-max before weighting.
- `employment_rank` was INVERTED (employment_deprivation = max_rank + 1 - rank) so higher value = more deprived = higher demand.
- All other features are positively associated with policing demand.

## Sensitivity Analysis
- Mean top-10% rank stability across all +-10% perturbations: **99.0%**
- Mean bottom-10% rank stability: **99.6%**
- High stability confirms the index is robust to small changes in weights.

## Literature Cross-Reference
- Laufs et al. (2021) identify crime volume, severity, and deprivation as the primary drivers of police demand -- consistent with high weights on severity_weighted_count and employment_deprivation.
- What Works Centre for Crime Reduction emphasises temporal volatility and footfall as demand amplifiers -- consistent with seasonal_volatility and total_footfall weights.
- Cambridge Crime Harm Index (Sherman et al., 2016) underpins the severity_weighted_count feature.