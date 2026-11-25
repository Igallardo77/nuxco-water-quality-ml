# Revision correspondence

This repository implements the November 2026 analytical revision of the 2016-2017 Nuxco record. It supersedes the original script and notebook without changing the original measured values.

| Review topic | Implementation / evidence |
|---|---|
| Sampling chronology and irregular spacing | `analysis_revision.py`, `gee_revision.py`; monthly distances and source reconciliation |
| Small-cluster inference and multiplicity | `gee_revision.py`, `gee_all_results.csv`, `numerical_validation.json` |
| Residuals, collinearity, suitability | `diagnostics_revision.py`, `extended_checks.py`; diagnostic JSON/CSV outputs |
| PCA and cluster stability | `stability_extended.json`, campaign-block and temporal-shift checks |
| Validation and selection | `nested_validation.json`, `nested_oof.csv`, held-out predictions |
| Anomaly influence | subset GEE fits and prediction sensitivity, `anomaly_flags.csv` |
| Nitrite and zero sensitivity | `zero_sensitivity.json`, `diagnostic_results.json` |
| Figure clarity and original layout | `manuscript_figures.py`, `figures/manuscript/` |

The manuscript's Figure 1 map is a supplied image, not regenerated from the analysis. Figures 2-10 are recreated by `manuscript_figures.py`; the diagnostic plots are additional reproducibility outputs.

The remaining field/laboratory records and physical zone justification are evidence gaps, not solved by statistical sensitivity. Some abstract clarifications were proposed in response letters, and the Vences-Martínez (2025) continuity explanation still needs explicit manuscript integration. The repository does not claim to have edited those documents.
