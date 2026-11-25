# Nuxco Lagoon: a historical physicochemical baseline (2016-2017)

Reproducible code and numerical results for **A Historical Physicochemical Baseline for Nuxco Lagoon Mexico Using Longitudinal Modeling and Interpretable Machine Learning**.

Five stations, thirteen campaigns from February 2016 to March 2017, and thirteen physicochemical variables yield 65 station-campaign profiles and 845 measurements. **No sampling was conducted in March 2016.** Dates are month labels; exact sampling days are unavailable. The workbook is included in `data/`.

This revision supersedes the earlier analysis and notebook. It treats the observations as a historical baseline, not a description of current lagoon conditions. Vences-Martínez et al. (2025) belongs to the team's continuing research; novelty is framed around the analytical questions and workflow rather than an absolute absence of recent studies.

## Reproduce

Validated with Python 3.12.14 and the exact versions in `requirements.txt`:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -r requirements.txt
python src/water_quality_analysis.py
```

The complete run includes resampling, nested model selection and high-resolution figures; allow several minutes. Outputs are written to `revision_2016_2017/results/`, `revision_2016_2017/figures/`, and `figures/manuscript/`. Existing outputs in these directories are regenerated. No source measurements are overwritten.

For Google Colab, open `notebooks/Nuxco_Water_Quality_Analysis.ipynb`. The notebook clones this repository using your GitHub access. For a private repository, grant access interactively; never insert tokens in notebook cells or commit credentials. The notebook invokes the same revised entry point.

## Revised workflow

- Source-sheet reconciliation: all 845 measurements compared with the original variable worksheets.
- Gaussian identity-link GEE with AR(1) correlation based on elapsed calendar-month distances; Mancl-DeRouen covariance correction; approximate t reference with 4 degrees of freedom and alternative 3-df interaction sensitivity.
- Benjamini-Hochberg correction across the thirteen group-by-time interactions within each specification.
- Refits excluding consensus anomalies, October-November campaigns, and individual stations; comparison with campaign-order correlation.
- Residual diagnostics, numerical covariance verification, collinearity, KMO/Bartlett assessment, parallel analysis and campaign-block resampling of PCA/clustering.
- Leave-one-campaign-out and leave-one-station-out evaluation; nested campaign-grouped validation with preprocessing and model selection inside training folds.
- Held-out permutation importance, classifier calibration, errors by campaign, nitrite-removal and recorded-zero sensitivities.
- Consensus anomaly screening with Isolation Forest, LOF and robust Mahalanobis distance, including eighteen alternative detector settings.

## Main revised results

| Analysis | Result | Scope |
|---|---|---|
| Corrected GEE interactions | Seven q < 0.05: BOD5, COD, TSS, settleable solids, ammonia nitrogen, nitrite, nitrate | Approximate inference with only five independent stations |
| DO group-by-time interaction | q = 0.1188; B-A slope difference 0.0458 mg/L/month; 95% CI [-0.0068, 0.0984] | Different DO slopes are not statistically established |
| PCA | PC1-PC2 75.0%; PC1-PC3 85.6% | Retention diagnostics support one or two axes depending on the temporal null; PC3 is unstable |
| Logistic classification | Balanced accuracy 0.8846 | Discriminates predefined station groups; does not validate ecological zonation |
| Fixed Elastic Net DO | R2 = 0.3847; RMSE = 0.4479 mg/L | Fixed-configuration campaign-held-out evaluation |
| Nested DO model selection | R2 = 0.2229; RMSE = 0.5033 mg/L | Selection of model family and settings occurs inside training |
| Anomalies | Six consensus flags; four October-November profiles persist in all 18 settings | Candidates for field/laboratory record review |

## Files

- `data/Base_Datos_Calidad_Agua_Integrada.xlsx`: original workbook, analysis sheet `01_Dataset_ML`.
- `src/water_quality_analysis.py`: complete-run entry point.
- `revision_2016_2017/`: revised modules, executed numerical results and detailed Spanish review report.
- `figures/manuscript/`: final manuscript panel layouts in PNG and SVG (Figure 1 map is supplied as PNG).
- `notebooks/Nuxco_Water_Quality_Analysis.ipynb`: revised Colab launcher.
- `docs/methodology_notes.md`: assumptions and unresolved documentation.
- `docs/revision_summary.md`: change record and manuscript correspondence.
- `docs/data_dictionary.md`: units and identifiers.
- `CITATION.cff`: author attribution; no publication DOI is asserted.

## Interpretation and outstanding evidence

The 65 profiles are repeated observations at **five independent stations**. Small-cluster corrections do not remove that limitation. Groups A (P1-P3) and B (P4-P5) are predefined; hydrological attribution requires independent information. Held-out-campaign evaluation uses earlier and later campaigns and is not prospective forecasting. Correlations and model importance do not establish causation. The DO model is exploratory, and does not replace direct field measurement or constitute an operational warning system.

Outstanding laboratory evidence includes confirmation of transport conditions, instrument/container records, accreditation, duplicates/recoveries, and the analytical meaning of recorded zeros and nitrogen reporting units. Hypothetical half-limit substitution cannot establish the meaning of a zero. See the methodology notes for the distinction between documented methods and unavailable records.

The repository update does not modify the submitted manuscript or resolve the remaining text-to-response alignment items. No reviewer correspondence or cover letters are included in the code distribution. Existing repository visibility is retained. Licensing and a persistent archive DOI remain subject to the authors' decisions.
