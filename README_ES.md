# Calidad del agua en la Laguna de Nuxco: estadística longitudinal y ML interpretable

Repositorio científico para reproducir el análisis fisicoquímico del agua en la **Laguna de Nuxco, Guerrero (México)**. El diseño contiene cinco puntos observados en trece campañas: **65 registros y 13 variables**.

## ¿Qué hace el código?

1. Verifica la estructura de los datos, calcula estadísticas y correlaciones.
2. Ajusta modelos GEE con correlación AR(1) y modelos mixtos complementarios.
3. Ejecuta PCA y clustering (K-means y Ward) para describir gradientes y perfiles.
4. Evalúa la clasificación exploratoria de dos grupos definidos previamente, P1–P3 y P4–P5.
5. Predice oxígeno disuelto mediante Ridge, Elastic Net, SVR y Random Forest, frente a un baseline.
6. Detecta muestras anómalas mediante Isolation Forest, LOF y Mahalanobis.
7. Calcula importancia fuera de muestra y SHAP opcional.
8. Exporta tablas y figuras a Excel/ZIP.

## Uso mediante Google Colab

Abre [Google Colab](https://colab.research.google.com/), selecciona **Archivo → Subir cuaderno** y carga `notebooks/Nuxco_Water_Quality_Analysis.ipynb`. Ejecuta todas las celdas y, cuando lo solicite, sube `data/Base_Datos_Calidad_Agua_Integrada.xlsx`. Al finalizar obtendrás el ZIP con gráficos, modelos y tablas de resultados.

## Uso desde Python

```bash
python -m pip install -r requirements.txt
python src/water_quality_analysis.py
```

Ejecuta desde la carpeta principal del repositorio. El programa busca la base en `data/` y crea una carpeta `resultados_calidad_agua/` y un archivo `Resultados_Calidad_Agua.zip`.

## Alcance científico

La clasificación supervisada no descubre dos zonas: estas se definieron previamente. Los tres clústeres son perfiles estadísticos y no niveles regulatorios de contaminación. Solo existen cinco sitios independientes: la inferencia GEE debe ser exploratoria. Las anomalías no prueban contaminación y SHAP no establece causalidad. Los sólidos sedimentables aparecen como `SS_mL_L` en los datos, mientras que una versión del manuscrito los presenta como mg/L: debe confirmarse la unidad con los resultados de laboratorio.

Para documentación, dependencias y limitaciones completas, consulta **[README.md](README.md)** (inglés).