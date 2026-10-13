# Methodological notes and limitations

- **Repeated observations:** 65 samples do not represent 65 independent sites. Gaussian GEE with working AR(1) is exploratory; only five independent clusters make standard sandwich errors and p-values potentially anti-conservative. Mixed models may have singular random-effect covariance.
- **Time axis:** `Orden_temporal` indexes 13 observed campaigns, not uniform elapsed months (March 2016 is missing). Do not interpret the coefficient as a constant per-calendar-month change without refitting a model using elapsed months.
- **PCA:** standardize features. Main analysis retains electrical conductivity while excluding redundant total dissolved solids and salinity.
- **Clustering:** K-means and Ward identify physicochemical profiles, not validated quality classes. Agreement between algorithms does not substitute for stability analysis.
- **Classification:** Zone A (P1–P3) and Zone B (P4–P5) are pre-assigned labels, not discovered ecological regions. Leave-one-period-out uses both earlier and later campaigns for training and is **not** prospective forecasting.
- **Prediction:** compare dissolved-oxygen MAE, RMSE and R² to the training-mean baseline, using identical held-out campaigns.
- **Anomalies:** unusual multivariate profiles are review candidates, not proven pollution events.
- **Importance:** held-out permutation importance is primary; SHAP fitted on full data is complementary.
- **Outstanding checks:** finite-cluster GEE corrections/sensitivity, correlation-structure comparisons, residual diagnostics, PCA suitability and bootstrap stability, probability calibration, independent site-zone justification, and laboratory settleable-solids units.
