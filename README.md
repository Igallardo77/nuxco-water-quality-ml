# Nuxco Lagoon Water Quality: Longitudinal Statistics and Interpretable Machine Learning

A reproducible and locally tested Python workflow for physicochemical water-quality monitoring in **Nuxco Lagoon, Guerrero, Mexico**.

This repository accompanies the manuscript draft *Longitudinal Modeling and Interpretable Machine Learning of Physicochemical Gradients in Nuxco Lagoon, Mexico*. It combines repeated-measures modeling and classical machine learning to describe water-quality gradients, identify multivariate profiles, evaluate predefined site-group separability, estimate dissolved oxygen, and flag atypical samples.

**Study design:** 5 fixed sampling sites × 13 campaigns (February 2016–March 2017, excluding March 2016) = **65 site-campaign observations**, with **13 physicochemical parameters**. The analysis dataset is stored in worksheet `01_Dataset_ML`.

> **Scientific scope:** exploratory research with only five independent sites. P1–P3 (“Zone A”) and P4–P5 (“Zone B”) are predefined analytical groups, not independently validated ecological zones. Anomalies are unusual observations, not proof of pollution.

## Repository structure

```text
nuxco-water-quality-ml/
├── README.md                         # English project documentation
├── README_ES.md                      # Spanish project overview
├── CITATION.cff                       # Citation metadata (DOI pending)
├── requirements.txt                  # Python dependencies
├── .gitignore
├── src/
│   └── water_quality_analysis.py     # Executable Python workflow
├── notebooks/
│   └── Nuxco_Water_Quality_Analysis.ipynb  # Google Colab notebook
├── data/
│   └── Base_Datos_Calidad_Agua_Integrada.xlsx
└── docs/
    ├── Nuxco_Manuscript_Preprint.pdf # Manuscript draft; approval needed before public release
    ├── methodology_notes.md         # Assumptions, limitations, unit discrepancy
    ├── data_dictionary.md           # Variable names and measurement units
    └── how_to_publish.md            # GitHub publication instructions
```

## Analytical workflow

| Stage | Methods | Output |
|---|---|---|
| Data quality and EDA | Missingness, uniqueness, descriptive statistics, Spearman correlations | Descriptive tables, heatmap |
| Longitudinal modeling | Gaussian GEE with AR(1) working correlation; random-intercept MixedLM sensitivity | Coefficients, standard errors, p-values; FDR for GEE zone×time interactions |
| Dimensionality reduction | Standardized PCA | Explained variance, scores, loadings |
| Unsupervised learning | K-means (k=2–4), Ward, silhouette and adjusted Rand index | Clusters, evaluation plots, dendrogram |
| Exploratory site-group classification | L2 logistic regression, linear SVM, shallow random forest | Out-of-fold predictions, balanced accuracy, macro-F1, ROC-AUC |
| Dissolved oxygen estimation | Ridge, Elastic Net, SVR, random forest | Out-of-fold MAE, RMSE, R² and baseline comparison |
| Anomaly screening | Isolation Forest, LOF, robust Mahalanobis distance | Consensus anomaly candidates |
| Interpretation | Out-of-sample permutation importance; optional SHAP | Ranked features, explanatory figures |

## Quick start: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and choose `notebooks/Nuxco_Water_Quality_Analysis.ipynb`.
3. Select **Runtime → Run all**.
4. When prompted, upload `data/Base_Datos_Calidad_Agua_Integrada.xlsx` (or upload it to the Colab file explorer before running).
5. Wait for the models and figures to finish. A ZIP archive called `Resultados_Calidad_Agua.zip` will be downloaded in Colab.

**Optional SHAP:** change `EJECUTAR_SHAP = False` to `True` in the notebook. Full-data SHAP figures are complementary, not evidence of out-of-sample performance.

## Quick start: local Python

Use **Python 3.10 or newer**. Open a terminal from the extracted repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/water_quality_analysis.py
```

The script automatically loads `data/Base_Datos_Calidad_Agua_Integrada.xlsx` from the repository. Results are written to `resultados_calidad_agua/`, and a zipped copy is created as `Resultados_Calidad_Agua.zip`.

Run optional SHAP from the shell with:

```bash
# Linux/macOS:
ENABLE_SHAP=1 python src/water_quality_analysis.py
# PowerShell:
# $env:ENABLE_SHAP="1"; python src/water_quality_analysis.py
```

## Inputs and reproducibility

The workbook contains one consolidated table of 65 rows, original per-variable worksheets, a data dictionary, and methodological planning sheets. The analysis uses **only** `01_Dataset_ML`, avoiding double-counting the original/transposed tables. See [the data dictionary](docs/data_dictionary.md).

Training/validation uses **leave-one-period-out** (13 held-out campaigns) as the primary strategy and **leave-one-site-out** as a secondary stress test. Out-of-fold results are evaluated using the same folds for all candidate algorithms. Predictive DO models are compared against a training-mean baseline, not judged by R² alone.

The code uses a fixed random seed (`42`) and saves its figures, detailed model results, predictions and anomaly candidates. The main Python script was smoke-tested successfully using the bundled 65-row workbook (without optional SHAP). Version-sensitive library outputs may vary; record Python and package versions for a publication-quality reproduction.

## Outputs

The main results workbook (`Resultados_Estadistica_ML_Calidad_Agua.xlsx`) contains data summaries, correlations, longitudinal model coefficients, PCA loadings and scores, clusters, grouped-validation metrics, predictions, permutation importance, and anomaly candidates. Figures are saved to `resultados_calidad_agua/figuras/` and bundled with the workbook in a ZIP archive.

## Interpretation limitations

- With **five sites**, GEE sandwich standard errors and asymptotic confidence intervals may be unreliable. Treat inferential p-values and q-values as exploratory pending small-cluster corrections and sensitivity checks.
- AR(1) is indexed by observed campaigns, **not** elapsed calendar days (one month is absent).
- The PCA components and clusters describe statistical patterns, **not causality** or regulatory water-quality classes.
- Group labels were defined from site IDs and need independent hydrological/geographical justification.
- Held-out-campaign validation is **not a truly prospective forecast** because the training data include campaigns before and after the held-out one.
- Consensus anomalies must be investigated against laboratory quality assurance and field/hydrometeorological records.
- Permutation importance from held-out predictions is primary; SHAP on the full dataset is supplementary.
- **Important measurement-unit check:** the workbook uses `SS_mL_L` for settleable solids; one manuscript version says mg/L. Verify the laboratory report before submission. These units are not interchangeable.

See [methodology notes](docs/methodology_notes.md) for full details.

## Scientific attribution and public release

Author names are listed in [`CITATION.cff`](CITATION.cff). The associated manuscript and supplementary code/data should be cited together once the article is published. No journal DOI or GitHub URL is claimed before assignment.

**Before making the repository public**, confirm approval from all data owners/coauthors, verify variable units and bibliographic metadata, and select explicit licenses for code, data and figures. This package intentionally does **not** assert rights or specify a license on the authors’ behalf.

Instructions for uploading to GitHub: [docs/how_to_publish.md](docs/how_to_publish.md).