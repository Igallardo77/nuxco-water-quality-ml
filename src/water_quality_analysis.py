# Código exportado del cuaderno de análisis de calidad del agua
# Ejecutar preferentemente en Google Colab.

# 
# # Calidad fisicoquímica del agua: estadística y machine learning interpretable
# 
# Este cuaderno utiliza la hoja **`01_Dataset_ML`** de la base integrada y ejecuta:
# 
# 1. Control de calidad y análisis descriptivo.
# 2. Modelos longitudinales: GEE con correlación AR(1) y modelo mixto de intercepto aleatorio.
# 3. PCA.
# 4. Clustering jerárquico y K-means.
# 5. Clasificación exploratoria de zonas P1–P3 frente a P4–P5.
# 6. Predicción del oxígeno disuelto.
# 7. Detección de anomalías por consenso.
# 8. Importancia de variables fuera de muestra y SHAP opcional.
# 9. Exportación de resultados a Excel y ZIP.
# 
# > **Diseño:** 5 puntos de muestreo × 13 periodos = 65 observaciones.  
# > **Precaución:** la zonificación P1–P3 y P4–P5 es exploratoria y debe justificarse con información geográfica, hidrológica o de fuentes de descarga.
# 


# Install dependencies using: python -m pip install -r requirements.txt


import os
import sys
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import chi2

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Autoregressive
from statsmodels.stats.multitest import multipletests

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, IsolationForest
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = Path("resultados_calidad_agua")
FIG_DIR = RESULTS_DIR / "figuras"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

print("Entorno preparado.")

# 
# ## 1. Cargar la base integrada
# 
# El código busca primero el archivo en `/content`, luego en `/mnt/data` y finalmente en la carpeta actual.  
# En Colab, si no lo encuentra, abrirá el selector para subirlo.
# 


NOMBRE_ARCHIVO = "Base_Datos_Calidad_Agua_Integrada.xlsx"

candidatos = [
    Path("/content") / NOMBRE_ARCHIVO,
    Path("/mnt/data") / NOMBRE_ARCHIVO,
    Path(NOMBRE_ARCHIVO),
    Path("data") / NOMBRE_ARCHIVO,
    Path(__file__).resolve().parent.parent / "data" / NOMBRE_ARCHIVO,
    Path("/mnt/data/Calidad_Agua_Integrada_Estadistica_ML.xlsx"),
]

ruta = next((p for p in candidatos if p.exists()), None)

if ruta is None and "google.colab" in sys.modules:
    from google.colab import files
    cargados = files.upload()
    if not cargados:
        raise FileNotFoundError("No se cargó ningún archivo.")
    ruta = Path(next(iter(cargados.keys())))

if ruta is None:
    raise FileNotFoundError(
        f"No se encontró {NOMBRE_ARCHIVO}. Colócalo junto al cuaderno o súbelo a Colab."
    )

df = pd.read_excel(ruta, sheet_name="01_Dataset_ML", engine="openpyxl")
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Punto"] = df["Punto"].astype(int)
df["Orden_temporal"] = df["Orden_temporal"].astype(int)
df["Periodo"] = df["Periodo"].astype(str)
df["Zona_binaria"] = (df["Zona_exploratoria"] == "Zona A (P1-P3)").astype(int)

print(f"Archivo: {ruta}")
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
display(df.head())


# Definición de variables
VARIABLES_AGUA = [
    "Temperatura_C", "pH", "CE_mS_cm", "SDT_mg_L", "Salinidad_UPS",
    "OD_mg_L", "DBO5_mg_L", "DQO_mg_L", "SST_mg_L", "SS_mL_L",
    "NH3_N_mg_L", "NO2_mg_L", "NO3_mg_L"
]

# Conjunto principal sin triplicar el gradiente de mineralización.
# Se conserva CE y se excluyen SDT y Salinidad en PCA/modelos predictivos principales.
FEATURES_PRINCIPALES = [
    "Temperatura_C", "pH", "CE_mS_cm", "OD_mg_L", "DBO5_mg_L", "DQO_mg_L",
    "SST_mg_L", "SS_mL_L", "NH3_N_mg_L", "NO2_mg_L", "NO3_mg_L"
]

FEATURES_OD = [x for x in FEATURES_PRINCIPALES if x != "OD_mg_L"]
VARIABLES_LOG = {
    "DBO5_mg_L", "DQO_mg_L", "SST_mg_L", "SS_mL_L",
    "NH3_N_mg_L", "NO2_mg_L", "NO3_mg_L"
}

assert len(df) == 65, "Se esperaban 65 registros."
assert df["ID_muestra"].is_unique, "Existen ID de muestra duplicados."
assert df[VARIABLES_AGUA].isna().sum().sum() == 0, "Existen valores faltantes."
assert df["Punto"].nunique() == 5, "Se esperaban cinco puntos."
assert df["Periodo"].nunique() == 13, "Se esperaban trece periodos."

print("Validación estructural correcta.")
print(df["Zona_exploratoria"].value_counts())

# ## 2. Control de calidad y descripción


descriptivos = df[VARIABLES_AGUA].agg(
    ["count", "mean", "std", "median", "min", "max"]
).T
descriptivos["CV_porcentaje"] = 100 * descriptivos["std"] / descriptivos["mean"]
descriptivos["faltantes"] = df[VARIABLES_AGUA].isna().sum()
descriptivos = descriptivos.reset_index(names="Variable")

resumen_punto = (
    df.groupby("Punto")[VARIABLES_AGUA]
      .agg(["mean", "std", "median", "min", "max"])
)

resumen_zona = (
    df.groupby("Zona_exploratoria")[VARIABLES_AGUA]
      .agg(["mean", "std", "median", "min", "max"])
)

correlaciones = df[VARIABLES_AGUA].corr(method="spearman")

display(descriptivos.round(3))


# Mapa de correlaciones sin seaborn
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(correlaciones.values, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(correlaciones.columns)))
ax.set_xticklabels(correlaciones.columns, rotation=90)
ax.set_yticks(range(len(correlaciones.index)))
ax.set_yticklabels(correlaciones.index)
ax.set_title("Correlaciones de Spearman")
fig.colorbar(im, ax=ax, label="rho")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_correlaciones_spearman.png", dpi=300, bbox_inches="tight")
plt.show()

# 
# ## 3. Modelos longitudinales
# 
# Se ajustan dos especificaciones para cada componente:
# 
# - **GEE AR(1):** reconoce correlación temporal dentro de cada punto.
# - **MixedLM:** modelo mixto con intercepto aleatorio por punto, como análisis de sensibilidad.
# 
# Debido a que solo existen cinco puntos, la inferencia entre zonas debe presentarse como exploratoria.  
# Se utiliza una tendencia temporal parsimoniosa (`Orden_temporal`) para evitar un modelo excesivamente parametrizado.
# 


def preparar_respuesta(datos, variable):
    temp = datos.copy()
    if variable in VARIABLES_LOG:
        temp["Y_modelo"] = np.log1p(temp[variable].clip(lower=0))
        escala = "log1p"
    else:
        temp["Y_modelo"] = temp[variable]
        escala = "original"
    return temp, escala


def extraer_resultados_ajuste(ajuste, variable, modelo, escala):
    filas = []
    parametros = ajuste.params
    errores = ajuste.bse
    pvals = ajuste.pvalues
    for termino in parametros.index:
        filas.append({
            "Variable": variable,
            "Modelo": modelo,
            "Escala": escala,
            "Termino": termino,
            "Estimacion": parametros.get(termino, np.nan),
            "Error_estandar": errores.get(termino, np.nan),
            "p": pvals.get(termino, np.nan),
        })
    return filas


resultados_modelos = []
errores_modelos = []

for variable in VARIABLES_AGUA:
    datos_m, escala = preparar_respuesta(df, variable)
    formula = "Y_modelo ~ C(Zona_exploratoria) * Orden_temporal"

    # GEE con AR(1)
    try:
        gee = smf.gee(
            formula=formula,
            groups="Punto",
            time="Orden_temporal",
            data=datos_m,
            family=sm.families.Gaussian(),
            cov_struct=Autoregressive()
        ).fit(maxiter=500)
        resultados_modelos.extend(
            extraer_resultados_ajuste(gee, variable, "GEE_AR1", escala)
        )
    except Exception as e:
        errores_modelos.append({
            "Variable": variable, "Modelo": "GEE_AR1", "Error": str(e)
        })

    # Modelo mixto con intercepto aleatorio
    try:
        mixed = smf.mixedlm(
            formula=formula,
            data=datos_m,
            groups=datos_m["Punto"],
            re_formula="1"
        ).fit(reml=False, method="lbfgs", maxiter=300, disp=False)
        resultados_modelos.extend(
            extraer_resultados_ajuste(mixed, variable, "MixedLM", escala)
        )
    except Exception as e:
        errores_modelos.append({
            "Variable": variable,
            "Modelo": "MixedLM",
            "Error": str(e)
        })

modelos_long = pd.DataFrame(resultados_modelos)
errores_modelos = pd.DataFrame(errores_modelos)

# FDR entre pruebas no relacionadas con el intercepto o la varianza aleatoria
modelos_long["q_FDR"] = np.nan
mask = (
    modelos_long["p"].notna()
    & ~modelos_long["Termino"].str.contains("Intercept|Group Var", regex=True)
)
if mask.any():
    modelos_long.loc[mask, "q_FDR"] = multipletests(
        modelos_long.loc[mask, "p"], method="fdr_bh"
    )[1]

display(modelos_long.round(5))
if not errores_modelos.empty:
    display(errores_modelos)

# ## 4. PCA


X_pca = df[FEATURES_PRINCIPALES].copy()
escalador_pca = StandardScaler()
X_pca_z = escalador_pca.fit_transform(X_pca)

pca = PCA()
scores = pca.fit_transform(X_pca_z)

varianza_pca = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Varianza_explicada": pca.explained_variance_ratio_,
    "Varianza_acumulada": np.cumsum(pca.explained_variance_ratio_)
})

cargas_pca = pd.DataFrame(
    pca.components_.T,
    index=FEATURES_PRINCIPALES,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
).reset_index(names="Variable")

scores_pca = df[
    ["ID_muestra", "Periodo", "Punto", "Zona_exploratoria"]
].copy()
for i in range(scores.shape[1]):
    scores_pca[f"PC{i+1}"] = scores[:, i]

display(varianza_pca.head().round(4))
display(cargas_pca[["Variable", "PC1", "PC2", "PC3"]].round(4))


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)
ax.axhline(0.80, linestyle="--")
ax.set_xlabel("Número de componentes")
ax.set_ylabel("Varianza acumulada")
ax.set_title("PCA: varianza explicada acumulada")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_pca_varianza.png", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
for zona, grupo in scores_pca.groupby("Zona_exploratoria"):
    ax.scatter(grupo["PC1"], grupo["PC2"], label=zona, s=55, alpha=0.8)
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA de perfiles fisicoquímicos")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "03_pca_scores_zona.png", dpi=300, bbox_inches="tight")
plt.show()

# ## 5. Clustering


silhouettes = []
modelos_kmeans = {}

for k in range(2, 5):
    km = KMeans(n_clusters=k, n_init=100, random_state=RANDOM_STATE)
    etiquetas = km.fit_predict(X_pca_z)
    score = silhouette_score(X_pca_z, etiquetas)
    silhouettes.append({"k": k, "Silhouette": score})
    modelos_kmeans[k] = (km, etiquetas)

tabla_silhouette = pd.DataFrame(silhouettes)
mejor_k = int(tabla_silhouette.loc[tabla_silhouette["Silhouette"].idxmax(), "k"])
etiquetas_kmeans = modelos_kmeans[mejor_k][1]

ward = AgglomerativeClustering(n_clusters=mejor_k, linkage="ward")
etiquetas_ward = ward.fit_predict(X_pca_z)

clustering = df[
    ["ID_muestra", "Periodo", "Punto", "Zona_exploratoria"]
].copy()
clustering["Cluster_KMeans"] = etiquetas_kmeans + 1
clustering["Cluster_Ward"] = etiquetas_ward + 1

ari = adjusted_rand_score(etiquetas_kmeans, etiquetas_ward)
nmi_zona = normalized_mutual_info_score(df["Zona_binaria"], etiquetas_kmeans)

print(f"Mejor k por silhouette: {mejor_k}")
print(f"ARI K-means vs Ward: {ari:.3f}")
print(f"NMI K-means vs zona exploratoria: {nmi_zona:.3f}")
display(tabla_silhouette.round(4))
display(pd.crosstab(clustering["Cluster_KMeans"], clustering["Zona_exploratoria"]))


Z = linkage(X_pca_z, method="ward")

fig, ax = plt.subplots(figsize=(13, 6))
dendrogram(
    Z,
    labels=df["ID_muestra"].tolist(),
    leaf_rotation=90,
    leaf_font_size=6,
    ax=ax
)
ax.set_title("Clustering jerárquico de Ward")
ax.set_ylabel("Distancia")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_dendrograma_ward.png", dpi=300, bbox_inches="tight")
plt.show()

# 
# ## 6. Clasificación exploratoria de zonas
# 
# - Clase 1: Zona A, puntos P1–P3.
# - Clase 0: Zona B, puntos P4–P5.
# - Validación principal: **Leave-One-Period-Out**.
# - Validación de estrés: **Leave-One-Point-Out**.
# 


def probabilidades_modelo(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    decision = modelo.decision_function(X)
    return 1 / (1 + np.exp(-decision))


def clasificacion_oof(modelo, X, y, grupos):
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y), dtype=float)
    fold = np.empty(len(y), dtype=object)

    for train_idx, test_idx in logo.split(X, y, grupos):
        m = clone(modelo)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        prob[test_idx] = probabilidades_modelo(m, X.iloc[test_idx])
        pred[test_idx] = (prob[test_idx] >= 0.5).astype(int)
        fold[test_idx] = str(grupos.iloc[test_idx[0]])

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metricas = {
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
        "F1_macro": f1_score(y, pred, average="macro"),
        "ROC_AUC": roc_auc_score(y, prob),
        "Sensibilidad_Zona_A": tp / (tp + fn) if (tp + fn) else np.nan,
        "Especificidad_Zona_B": tn / (tn + fp) if (tn + fp) else np.nan,
    }
    preds = pd.DataFrame({
        "ID_muestra": df["ID_muestra"],
        "Real": y,
        "Prediccion": pred,
        "Prob_Zona_A": prob,
        "Fold": fold
    })
    return metricas, preds


X_clf = df[FEATURES_PRINCIPALES]
y_clf = df["Zona_binaria"]

modelos_clf = {
    "Logistica_L2": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ]),
    "SVM_lineal": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear", C=1.0, probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        ))
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=300, max_depth=3, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE
    )
}

metricas_clf = []
predicciones_clf = []

for esquema, grupos in {
    "Leave_One_Period_Out": df["Periodo"],
    "Leave_One_Point_Out": df["Punto"]
}.items():
    for nombre, modelo in modelos_clf.items():
        met, pred = clasificacion_oof(modelo, X_clf, y_clf, grupos)
        met.update({"Esquema": esquema, "Modelo": nombre})
        metricas_clf.append(met)
        pred["Esquema"] = esquema
        pred["Modelo"] = nombre
        predicciones_clf.append(pred)

metricas_clf = pd.DataFrame(metricas_clf)
predicciones_clf = pd.concat(predicciones_clf, ignore_index=True)

display(
    metricas_clf.sort_values(
        ["Esquema", "Balanced_accuracy"], ascending=[True, False]
    ).round(4)
)


# Matriz de confusión del mejor modelo en Leave-One-Period-Out
tabla_periodo = metricas_clf.query("Esquema == 'Leave_One_Period_Out'")
mejor_clf = tabla_periodo.sort_values("Balanced_accuracy", ascending=False).iloc[0]["Modelo"]

pred_mejor_clf = predicciones_clf.query(
    "Esquema == 'Leave_One_Period_Out' and Modelo == @mejor_clf"
)
cm = confusion_matrix(
    pred_mejor_clf["Real"], pred_mejor_clf["Prediccion"], labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Zona B", "Zona A"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Zona B", "Zona A"])
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión: {mejor_clf}")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "05_matriz_confusion_zonas.png", dpi=300, bbox_inches="tight")
plt.show()

# ### Importancia agrupada fuera de muestra para clasificación


def importancia_oof_clasificacion(modelo, X, y, grupos, n_repeticiones=30):
    logo = LeaveOneGroupOut()
    folds = []
    y_total = np.zeros(len(y), dtype=int)
    base_prob = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in logo.split(X, y, grupos):
        m = clone(modelo)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        base_prob[test_idx] = probabilidades_modelo(m, X.iloc[test_idx])
        y_total[test_idx] = y.iloc[test_idx].to_numpy()
        folds.append((m, test_idx))

    base_loss = log_loss(y_total, base_prob, labels=[0, 1])
    rng = np.random.default_rng(RANDOM_STATE)
    filas = []

    for feature in X.columns:
        incrementos = []
        for _ in range(n_repeticiones):
            perm_prob = base_prob.copy()
            for m, test_idx in folds:
                X_test = X.iloc[test_idx].copy()
                X_test[feature] = rng.permutation(X_test[feature].to_numpy())
                perm_prob[test_idx] = probabilidades_modelo(m, X_test)
            perm_loss = log_loss(y_total, perm_prob, labels=[0, 1])
            incrementos.append(perm_loss - base_loss)

        filas.append({
            "Variable": feature,
            "Importancia_media_DeltaLogLoss": np.mean(incrementos),
            "DE": np.std(incrementos, ddof=1)
        })

    return pd.DataFrame(filas).sort_values(
        "Importancia_media_DeltaLogLoss", ascending=False
    )


modelo_mejor_clf = modelos_clf[mejor_clf]
importancia_clf = importancia_oof_clasificacion(
    modelo_mejor_clf,
    X_clf,
    y_clf,
    df["Periodo"],
    n_repeticiones=30
)
display(importancia_clf.round(5))

fig, ax = plt.subplots(figsize=(8, 5))
orden = importancia_clf.sort_values("Importancia_media_DeltaLogLoss")
ax.barh(
    orden["Variable"],
    orden["Importancia_media_DeltaLogLoss"],
    xerr=orden["DE"]
)
ax.set_xlabel("Incremento de log-loss al permutar")
ax.set_title("Importancia fuera de muestra: clasificación de zonas")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_importancia_clasificacion.png", dpi=300, bbox_inches="tight")
plt.show()

# ## 7. Predicción del oxígeno disuelto


def regresion_oof(modelo, X, y, grupos):
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y), dtype=float)
    fold = np.empty(len(y), dtype=object)

    for train_idx, test_idx in logo.split(X, y, grupos):
        m = clone(modelo)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = m.predict(X.iloc[test_idx])
        fold[test_idx] = str(grupos.iloc[test_idx[0]])

    metricas = {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred)
    }
    preds = pd.DataFrame({
        "ID_muestra": df["ID_muestra"],
        "OD_real": y,
        "OD_predicho": pred,
        "Residuo": y - pred,
        "Fold": fold
    })
    return metricas, preds


X_od = df[FEATURES_OD]
y_od = df["OD_mg_L"]

modelos_od = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Elastic_Net": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=0.05, l1_ratio=0.5, max_iter=20000,
            random_state=RANDOM_STATE
        ))
    ]),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=10.0, epsilon=0.10, gamma="scale"))
    ]),
    "Random_Forest": RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=3,
        random_state=RANDOM_STATE
    )
}

metricas_od = []
predicciones_od = []

for esquema, grupos in {
    "Leave_One_Period_Out": df["Periodo"],
    "Leave_One_Point_Out": df["Punto"]
}.items():
    for nombre, modelo in modelos_od.items():
        met, pred = regresion_oof(modelo, X_od, y_od, grupos)
        met.update({"Esquema": esquema, "Modelo": nombre})
        metricas_od.append(met)
        pred["Esquema"] = esquema
        pred["Modelo"] = nombre
        predicciones_od.append(pred)

metricas_od = pd.DataFrame(metricas_od)
predicciones_od = pd.concat(predicciones_od, ignore_index=True)

# Baseline: media del entrenamiento en cada fold
baseline = Pipeline([("model", Ridge(alpha=1e12))])
met_base, pred_base = regresion_oof(
    baseline, pd.DataFrame({"Constante": np.ones(len(df))}), y_od, df["Periodo"]
)
met_base.update({"Esquema": "Leave_One_Period_Out", "Modelo": "Baseline_media"})
metricas_od = pd.concat([metricas_od, pd.DataFrame([met_base])], ignore_index=True)

display(metricas_od.sort_values(["Esquema", "RMSE"]).round(4))


tabla_od_periodo = metricas_od.query(
    "Esquema == 'Leave_One_Period_Out' and Modelo != 'Baseline_media'"
)
mejor_od = tabla_od_periodo.sort_values("RMSE").iloc[0]["Modelo"]

pred_mejor_od = predicciones_od.query(
    "Esquema == 'Leave_One_Period_Out' and Modelo == @mejor_od"
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pred_mejor_od["OD_real"], pred_mejor_od["OD_predicho"], s=50, alpha=0.8)
lim_min = min(pred_mejor_od["OD_real"].min(), pred_mejor_od["OD_predicho"].min())
lim_max = max(pred_mejor_od["OD_real"].max(), pred_mejor_od["OD_predicho"].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
ax.set_xlabel("OD observado (mg/L)")
ax.set_ylabel("OD predicho (mg/L)")
ax.set_title(f"Predicción fuera de muestra: {mejor_od}")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_prediccion_od.png", dpi=300, bbox_inches="tight")
plt.show()

# ### Importancia agrupada fuera de muestra para predicción de OD


def importancia_oof_regresion(modelo, X, y, grupos, n_repeticiones=30):
    logo = LeaveOneGroupOut()
    folds = []
    base_pred = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in logo.split(X, y, grupos):
        m = clone(modelo)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        base_pred[test_idx] = m.predict(X.iloc[test_idx])
        folds.append((m, test_idx))

    base_rmse = mean_squared_error(y, base_pred) ** 0.5
    rng = np.random.default_rng(RANDOM_STATE)
    filas = []

    for feature in X.columns:
        incrementos = []
        for _ in range(n_repeticiones):
            perm_pred = base_pred.copy()
            for m, test_idx in folds:
                X_test = X.iloc[test_idx].copy()
                X_test[feature] = rng.permutation(X_test[feature].to_numpy())
                perm_pred[test_idx] = m.predict(X_test)
            perm_rmse = mean_squared_error(y, perm_pred) ** 0.5
            incrementos.append(perm_rmse - base_rmse)

        filas.append({
            "Variable": feature,
            "Importancia_media_DeltaRMSE": np.mean(incrementos),
            "DE": np.std(incrementos, ddof=1)
        })

    return pd.DataFrame(filas).sort_values(
        "Importancia_media_DeltaRMSE", ascending=False
    )


modelo_mejor_od = modelos_od[mejor_od]
importancia_od = importancia_oof_regresion(
    modelo_mejor_od,
    X_od,
    y_od,
    df["Periodo"],
    n_repeticiones=30
)
display(importancia_od.round(5))

fig, ax = plt.subplots(figsize=(8, 5))
orden = importancia_od.sort_values("Importancia_media_DeltaRMSE")
ax.barh(
    orden["Variable"],
    orden["Importancia_media_DeltaRMSE"],
    xerr=orden["DE"]
)
ax.set_xlabel("Incremento de RMSE al permutar")
ax.set_title("Importancia fuera de muestra: predicción de OD")
fig.tight_layout()
fig.savefig(FIG_DIR / "08_importancia_od.png", dpi=300, bbox_inches="tight")
plt.show()

# ## 8. Detección de anomalías por consenso


X_anom = df[FEATURES_PRINCIPALES]
X_anom_z = RobustScaler().fit_transform(X_anom)

# Isolation Forest
iforest = IsolationForest(
    n_estimators=500, contamination=0.10,
    random_state=RANDOM_STATE
)
pred_if = iforest.fit_predict(X_anom_z)
score_if = -iforest.score_samples(X_anom_z)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=15, contamination=0.10)
pred_lof = lof.fit_predict(X_anom_z)
score_lof = -lof.negative_outlier_factor_

# Distancia robusta de Mahalanobis
mcd = MinCovDet(random_state=RANDOM_STATE, support_fraction=0.80).fit(X_anom_z)
md2 = mcd.mahalanobis(X_anom_z)
umbral_md2 = chi2.ppf(0.90, df=X_anom_z.shape[1])
pred_md = np.where(md2 > umbral_md2, -1, 1)

anomalias = df[
    ["ID_muestra", "Periodo", "Punto", "Zona_exploratoria"]
].copy()
anomalias["Score_IsolationForest"] = score_if
anomalias["Score_LOF"] = score_lof
anomalias["Mahalanobis2"] = md2
anomalias["Anomalia_IF"] = (pred_if == -1).astype(int)
anomalias["Anomalia_LOF"] = (pred_lof == -1).astype(int)
anomalias["Anomalia_MD"] = (pred_md == -1).astype(int)
anomalias["Consenso"] = anomalias[
    ["Anomalia_IF", "Anomalia_LOF", "Anomalia_MD"]
].sum(axis=1)
anomalias["Anomalia_consenso"] = (anomalias["Consenso"] >= 2).astype(int)

anomalias = anomalias.sort_values(
    ["Anomalia_consenso", "Consenso", "Score_IsolationForest"],
    ascending=[False, False, False]
)

display(anomalias.head(15).round(4))


fig, ax = plt.subplots(figsize=(11, 5))
orden_plot = anomalias.sort_values(["Periodo", "Punto"])
x = np.arange(len(orden_plot))
ax.scatter(
    x,
    orden_plot["Score_IsolationForest"],
    s=35 + 45 * orden_plot["Anomalia_consenso"]
)
ax.set_xticks(x[::5])
ax.set_xticklabels(orden_plot["ID_muestra"].iloc[::5], rotation=90)
ax.set_ylabel("Score Isolation Forest")
ax.set_title("Puntuación de anomalía por muestra")
fig.tight_layout()
fig.savefig(FIG_DIR / "09_anomalias.png", dpi=300, bbox_inches="tight")
plt.show()

# 
# ## 9. SHAP opcional
# 
# La importancia fuera de muestra calculada arriba es el resultado interpretativo principal.  
# SHAP se presenta como complemento exploratorio después de validar el rendimiento.
# 
# Cambiar `EJECUTAR_SHAP = True` para generar las figuras.
# 


EJECUTAR_SHAP = False

if EJECUTAR_SHAP:
    import shap

    # Clasificación con Random Forest
    rf_clf = modelos_clf["Random_Forest"]
    rf_clf.fit(X_clf, y_clf)
    explainer_clf = shap.TreeExplainer(rf_clf)
    valores_clf = explainer_clf.shap_values(X_clf)

    if isinstance(valores_clf, list):
        valores_clf_plot = valores_clf[1]
    else:
        valores_clf_plot = np.asarray(valores_clf)
        if valores_clf_plot.ndim == 3:
            valores_clf_plot = valores_clf_plot[:, :, 1]

    shap.summary_plot(
        valores_clf_plot, X_clf,
        feature_names=FEATURES_PRINCIPALES,
        show=False
    )
    plt.title("SHAP exploratorio: clasificación de zonas")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_shap_clasificacion.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Regresión de OD con Random Forest
    rf_od = modelos_od["Random_Forest"]
    rf_od.fit(X_od, y_od)
    explainer_od = shap.TreeExplainer(rf_od)
    valores_od = explainer_od.shap_values(X_od)

    shap.summary_plot(
        valores_od, X_od,
        feature_names=FEATURES_OD,
        show=False
    )
    plt.title("SHAP exploratorio: predicción de OD")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_shap_od.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("SHAP omitido. Cambia EJECUTAR_SHAP a True para ejecutarlo.")

# ## 10. Exportar todos los resultados


salida_excel = RESULTS_DIR / "Resultados_Estadistica_ML_Calidad_Agua.xlsx"

with pd.ExcelWriter(salida_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Datos_integrados", index=False)
    descriptivos.to_excel(writer, sheet_name="Descriptivos", index=False)
    resumen_punto.to_excel(writer, sheet_name="Resumen_punto")
    resumen_zona.to_excel(writer, sheet_name="Resumen_zona")
    correlaciones.to_excel(writer, sheet_name="Correlaciones")
    modelos_long.to_excel(writer, sheet_name="Modelos_longitudinales", index=False)

    if not errores_modelos.empty:
        errores_modelos.to_excel(writer, sheet_name="Errores_modelos", index=False)

    varianza_pca.to_excel(writer, sheet_name="PCA_varianza", index=False)
    cargas_pca.to_excel(writer, sheet_name="PCA_cargas", index=False)
    scores_pca.to_excel(writer, sheet_name="PCA_scores", index=False)
    tabla_silhouette.to_excel(writer, sheet_name="Clustering_metricas", index=False)
    clustering.to_excel(writer, sheet_name="Clustering_muestras", index=False)
    metricas_clf.to_excel(writer, sheet_name="Clasificacion_metricas", index=False)
    predicciones_clf.to_excel(writer, sheet_name="Clasificacion_OOF", index=False)
    importancia_clf.to_excel(writer, sheet_name="Importancia_clasif", index=False)
    metricas_od.to_excel(writer, sheet_name="OD_metricas", index=False)
    predicciones_od.to_excel(writer, sheet_name="OD_predicciones_OOF", index=False)
    importancia_od.to_excel(writer, sheet_name="Importancia_OD", index=False)
    anomalias.to_excel(writer, sheet_name="Anomalias", index=False)

# Archivo de texto con las decisiones automáticas
resumen_txt = RESULTS_DIR / "Resumen_ejecucion.txt"
with open(resumen_txt, "w", encoding="utf-8") as f:
    f.write("ANÁLISIS DE CALIDAD FISICOQUÍMICA DEL AGUA\n")
    f.write("=" * 50 + "\n")
    f.write(f"Archivo analizado: {ruta}\n")
    f.write(f"Registros: {len(df)}\n")
    f.write(f"Mejor k de clustering: {mejor_k}\n")
    f.write(f"Mejor clasificador Leave-One-Period-Out: {mejor_clf}\n")
    f.write(f"Mejor regresor de OD Leave-One-Period-Out: {mejor_od}\n")
    f.write(
        f"Anomalías por consenso: "
        f"{int(anomalias['Anomalia_consenso'].sum())}\n"
    )
    f.write(
        "\nNota: la zona P1–P3 vs P4–P5 es exploratoria y requiere "
        "justificación ambiental independiente.\n"
    )

zip_path = shutil.make_archive(
    "Resultados_Calidad_Agua",
    "zip",
    root_dir=RESULTS_DIR
)

print(f"Resultados Excel: {salida_excel}")
print(f"Resultados ZIP: {zip_path}")

if "google.colab" in sys.modules:
    from google.colab import files
    files.download(zip_path)

# 
# ## Interpretación recomendada
# 
# 1. Primero interprete los modelos longitudinales y sus intervalos de confianza.
# 2. Use PCA y clustering para describir gradientes y perfiles, no para afirmar causalidad.
# 3. La clasificación de zonas solo es válida como análisis exploratorio mientras no exista una definición geográfica independiente.
# 4. Compare siempre los modelos predictivos contra el baseline.
# 5. Una anomalía representa rareza multivariada, no necesariamente contaminación.
# 6. Reporte la importancia fuera de muestra como resultado principal; SHAP es complementario.
# 
