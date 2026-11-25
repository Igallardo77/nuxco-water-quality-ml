# Nuxco: línea base fisicoquímica histórica, 2016-2017

Actualización del código y los resultados de la última revisión del manuscrito. El periodo correcto es febrero de 2016 a marzo de 2017; en marzo de 2016 no hubo muestreo. Se analizaron 65 perfiles de cinco estaciones, trece campañas y trece variables.

## Ejecución

```bash
python -m pip install -r requirements.txt
python src/water_quality_analysis.py
```

Entorno verificado: Python 3.12.14 y versiones fijadas en `requirements.txt`. El notebook de `notebooks/` ejecuta este mismo programa. Los resultados se regeneran en `revision_2016_2017/results/`, los diagnósticos en `revision_2016_2017/figures/` y las figuras finales en `figures/manuscript/`.

## Cambios principales

Se incorporaron meses calendario, covarianza GEE Mancl-DeRouen, referencias t para pocos sitios, corrección BH de trece interacciones, diagnósticos, remuestreo, validación anidada por campaña y sensibilidad a anomalías, nitrito y ceros. Siete interacciones conservan q < 0.05; la interacción de oxígeno tiene q = 0.1188. El Elastic Net fijo obtiene R2 = 0.3847 y RMSE = 0.4479 mg/L; la selección anidada obtiene R2 = 0.2229 y RMSE = 0.5033 mg/L. Son evaluaciones distintas y se reportan por separado.

Se mantienen los tipos y la organización de las figuras revisadas. El estudio se interpreta como línea base histórica exploratoria; Vences-Martínez et al. (2025) forma parte de la continuidad del equipo. Los resultados no establecen condiciones actuales, causalidad ni una herramienta operativa validada.

La justificación física de los grupos A y B y algunos respaldos de campo/laboratorio siguen pendientes. La actualización del repositorio tampoco sustituye la integración de las precisiones pendientes en el manuscrito. Consulte `README.md`, `docs/methodology_notes.md` y el informe detallado `revision_2016_2017/Resultados_revision_Nuxco.md`.
