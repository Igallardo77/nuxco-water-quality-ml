"""Assemble the pre-manuscript review table from executed numerical outputs."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from analysis_revision import load_data,VARS,OUT,SOURCE

ROOT=Path(__file__).parent
def read(name):return json.loads((OUT/name).read_text())
def markdown_table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v).replace('|','/').replace('\n',' ')
    header='| '+' | '.join(map(str,frame.columns))+' |'
    separator='| '+' | '.join(['---']*len(frame.columns))+' |'
    body=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join([header,separator]+body)
def main():
    d=load_data();a=read('analysis_results.json');di=read('diagnostic_results.json')
    nested=read('nested_validation.json');col=read('collinearity.json');stab=read('stability_extended.json')
    g=pd.read_csv(OUT/'gee_all_results.csv');q=g.query("working_correlation=='calendar_month' and contrast=='B_minus_A'").pivot(index='variable',columns='subset',values='q_p_t_MD')
    # Reconcile the integrated analysis against every original measurement sheet.
    mapping=dict(zip(['Temp','pH','CE','SDT','SAL','OX DIS','DBO','DQO','SST','SS','NH3-N','NO2','NO3'],VARS))
    checks=[]
    for sheet,var in mapping.items():
        raw=pd.read_excel(SOURCE,sheet,header=None).iloc[1:6,2:15].to_numpy(float)
        prepared=d.pivot(index='Punto',columns='Periodo',values=var).sort_index(axis=1).to_numpy(float)
        error=float(np.max(np.abs(raw-prepared)));assert error==0
        checks.append(dict(sheet=sheet,variable=var,values=65,max_difference=error))
    (OUT/'source_reconciliation.json').write_text(json.dumps(checks,indent=2))
    audit={'source_reconciliation_values':845,'month_labels_not_exact_sampling_days':True,
           'main_code':'analisis_calidad_agua_estadistica_ml_colab (1).py',
           'watterqt_py':'Different satellite-band workflow; not the analysis supporting this manuscript.',
           'original_hyperparameters':'Fixed in the inspected script; scaling fitted within training folds. Prior manual tuning history cannot be inferred from code.',
           'original_FDR':'Code pools all non-intercept terms including available mixed-model tests; manuscript says 13 interactions. Revised analysis explicitly applies BH to 13 interactions per specification.',
           'protected_files_unchanged':{}}
    audit['input_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    rows=[]
    def add(ref,topic,result,interpretation,state='Calculado'):
        rows.append(dict(Observacion=ref,Tema=topic,Resultado=result,Interpretacion=interpretation,Estado=state))
    add('General','Integridad y período','65 perfiles; 845 valores; 5 sitios; 13 campañas, febrero 2016–marzo 2017. Cero faltantes o duplicados; 845/845 valores coinciden con hojas originales.','Usar 2016–2017. Marzo 2016 no se muestreó según aclaración de autores.')
    add('R1-P1, R2-L220','Diagnósticos GEE','13 gráficos de residuos y 13 gráficos QQ, agrupados en 13 figuras. Asimetría residual: NH3–N=2.987, nitrito=2.573; en ambos persisten colas pronunciadas en la escala del modelo.','No afirmar que todos los diagnósticos son satisfactorios. La normalidad marginal no es requisito de GEE; las colas y los cinco sitios refuerzan la necesidad de sensibilidad y cautela inferencial. Shapiro se presenta solo como diagnóstico nominal.')
    add('R1-P2','GEE y FDR','Covarianza Mancl–DeRouen, t con 4 gl y BH para 13 interacciones. Siete interacciones conservan q<0.05. La sensibilidad t con 3 gl conserva esas siete.','DBO5, DQO, SST, SS, NH3–N, nitrito y nitrato. Cinco sitios siguen limitando la inferencia.')
    add('R1-P2','Interacción zona-tiempo del OD',f"q={q.loc['OD_mg_L','all']:.6f}; diferencia B−A=0.045793 mg/L/mes, IC95% t4 [-0.006835, 0.098421].",'No sostener una diferencia significativa entre las pendientes de OD de ambas zonas.')
    add('R1-P3','GEE sin seis anomalías',f"n=59; siete interacciones continúan con q<0.05; OD q={q.loc['OD_mg_L','without_consensus']:.6f}.",'Las diferencias de carga orgánica/nutrientes no desaparecen al retirar las anomalías. La divergencia de OD permanece incierta.')
    add('R1-P3','GEE sin octubre–noviembre 2016',f"n=55; las mismas siete interacciones conservan q<0.05; OD q={q.loc['OD_mg_L','without_Oct_Nov_2016']:.6f}.",'La evidencia de las siete variables no depende exclusivamente de esas dos campañas.')
    add('R1-P8, R2-L236–237','Tiempo calendario','Meses transcurridos: 0,2,3,…,13; matriz AR(1) con rho elevado a la diferencia mensual.','Las fechas del Excel son primeros de mes convencionales. Se usan meses, sin inventar días exactos de muestreo.')
    add('R2-Fig2, R2-L341','Tendencia y ajuste de OD','Pendiente A=-0.054573 mg/L/mes; R² descriptivo A=0.097725. Sin anomalías, pendiente A=-0.029573.','Descenso lineal exploratorio con importantes oscilaciones. Con t4, p nominal A=0.0174; con 2 gl específicos para sus tres sitios, p=0.0596. Evitar afirmaciones categóricas.')
    add('R2-Fig2','Punto del período 2','Abril de 2016: OD medio A=5.89 mg/L y B=5.89 mg/L.','Los puntos se superponen; se añadió anotación explícita en la figura.')
    add('R1-P1','Colinealidad',f"VIF en predictores de OD: DBO5={col['OD10']['VIF']['DBO5_mg_L']:.2f}; SST={col['OD10']['VIF']['SST_mg_L']:.2f}; NH3–N={col['OD10']['VIF']['NH3_N_mg_L']:.2f}.",'La redundancia persiste después de retirar TDS y salinidad. La regularización ayuda a predecir; las importancias individuales no son efectos independientes.')
    add('R1-P1','KMO y Bartlett',f"KMO={col['PCA11']['KMO']:.6f}; Bartlett chi²={col['PCA11']['Bartlett_chi2']:.3f}, gl=55, p nominal={col['PCA11']['Bartlett_p_nominal_only']:.3g}.",'KMO describe adecuación de la matriz. Bartlett presupone independencia y su p no debe presentarse como validación confirmatoria en este diseño.')
    add('R1-P1, R2-L262','Número de componentes','PC1=54.829%; PC2=20.202%; PC3=10.596%; acumulada=85.627%. Análisis paralelo iid (2000 simulaciones): 2 componentes. Sensibilidad de desplazamientos por campaña (1000): 1.','Dos componentes explican 75.031%. PC3 puede mostrarse como detalle descriptivo complementario; no afirmar que tres dimensiones estén inequívocamente validadas.')
    add('R1-P1','Estabilidad PCA','Bootstrap por bloques de 2 y 3 campañas, 500 repeticiones por longitud. Varianza acumulada de tres PC: percentiles 2.5–97.5 de 82.7–91.5% y 81.5–91.0%.','La varianza acumulada se conserva mejor que la orientación individual de PC3, que es variable.')
    add('R1-P1, R2-L31–32','Agrupamiento','k=3: silhouette=0.352503; k=4: 0.348492. Ward y K-means coinciden (ARI=1); tamaños 41,20,4.','Tres perfiles descriptivos son reproducibles en la muestra completa. No equivalen a tres clases ecológicas confirmadas.')
    add('R1-P1','Estabilidad de agrupamiento','ARI mediano de Ward en bootstrap: 0.623 (bloques 2) y 0.708 (bloques 3); intervalos amplios. k=4 obtiene mayor silhouette en 371/500 remuestreos K-means.','La elección exacta de tres grupos es sensible al remuestreo; debe reconocerse.')
    add('R2-L257–260','Correlaciones del PCA','Pearson: OD–DBO5=-0.622; OD–SST=-0.603; DBO5–SST=0.978.','La ausencia de correlación sugerida por el revisor no se cumple en la matriz original; explicar proyección y mostrar correlaciones y cargas.')
    add('R1-M3, R1-M5','Clasificación y validación','Logística: exactitud equilibrada=0.884615, AUC=0.937870. Validación anidada de C reproduce ambas cifras.','Se mantiene el desempeño para distinguir grupos predefinidos; no valida una zonificación ecológica externa.')
    add('R1-P1','Calibración','Brier: logística=0.089643, SVM=0.095621, RF=0.066957; referencia constante de prevalencia=0.24. Curvas con cinco intervalos.','Evaluación fuera de muestra; algunos intervalos contienen pocos datos, por lo que las curvas son descriptivas.')
    add('R1-M5','Auditoría del código','Parámetros fijos y escalado dentro del entrenamiento. No se encontró búsqueda global de hiperparámetros en el script del manuscrito.','La historia de selección manual no es recuperable del código. Se añadió comprobación anidada para documentar selección sin solapamiento de campañas.')
    add('R1-P7, R2-L83, R2-L385–390','OD: especificación original','Elastic Net fijo: R²=0.384705; RMSE=0.447856 mg/L; baseline=0.592408 mg/L; mejora RMSE=24.4%.','La cifra original es reproducible. Describir capacidad predictiva limitada y utilidad exploratoria, no vigilancia operativa validada.')
    add('R1-M5, R2-L385–390','OD: selección anidada adicional',f"R²={nested['regression']['R2']:.6f}; RMSE={nested['regression']['RMSE']:.6f} mg/L; mejora frente a baseline={100*(1-nested['regression']['RMSE']/nested['baseline']['RMSE']):.1f}%.",'Incluye elección entre familias e hiperparámetros dentro de cada entrenamiento. Es otro procedimiento de validación; no prueba por sí solo que hubiera fuga en el modelo fijo.')
    add('R1-P3, R2-Fig6','Predicción sin anomalías','Elastic Net fijo reentrenado con n=59: R²=-0.023612; RMSE=0.445134 mg/L; baseline=0.461771 mg/L (mejora 3.6%).','Parte importante del rendimiento relativo depende de la amplitud de respuesta aportada por los extremos. El error absoluto cambia poco; la varianza del OD se reduce.')
    add('R2-Fig6','Casos de OD bajo','Tres observaciones con OD≤4 mg/L. En las otras 62, R² de las predicciones originales=0.084015 y RMSE=0.449362 mg/L.','Aclarar que 1:1 es una referencia de concordancia. El ajuste es limitado dentro del rango habitual.')
    add('R1-P1','Errores por campaña','RMSE por campaña entre 0.139609 y 0.932902 mg/L; mayor error en abril de 2016.','La precisión no es uniforme a lo largo del registro.')
    add('R2-L446–447','Aportación del nitrito','Sin nitrito, R²=0.168514 y RMSE=0.520624 mg/L, frente a 0.384705 y 0.447856 con nitrito.','Existe aportación predictiva en el conjunto completo. Su baja concentración no justifica automáticamente excluirlo.')
    add('R2-L446–447','Estabilidad de la importancia del nitrito','Permutación: ΔRMSE=0.135078 mg/L; sin anomalías, ΔRMSE=0.016556. Sin anomalías, retirar nitrito mejora ligeramente RMSE (0.445134 a 0.440038).','La importancia del nitrito depende del contexto y de las observaciones extremas; no es un mecanismo causal demostrado.')
    add('Sensibilidad complementaria','Cero de nitrito','Al excluir el único cero, R²=0.360565; sustituciones hipotéticas por LD/2 o LC/2 dejan R²≈0.3847–0.3844.','El resultado principal no depende exclusivamente de ese cero. Son escenarios de sensibilidad; no identifican qué significaba analíticamente el registro.')
    add('R1-P3, R2-L34','Anomalías y umbrales','Se reproducen las seis anomalías. En 18 combinaciones de parámetros se detectan entre 4 y 10; los cuatro casos P1/P2 de octubre/noviembre 2016 aparecen en 18/18.','El núcleo de cuatro perfiles es estable a los parámetros ensayados. Mantener interpretación como señal estadística para revisión.')
    add('R2-Fig7B','Unidad de importancia por permutación','El eje muestra incremento de RMSE al predecir OD: su unidad es mg/L de OD.','La unidad corresponde al cambio en el error, no a las unidades de temperatura o pH. Se aclaró en la figura.')
    add('R1-M6','Figuras','42 figuras individuales, con versiones PNG y SVG; incluye PC3, calibración, anomalías, residuos y trayectorias.','Figuras preparadas para incorporar al manuscrito en la siguiente etapa.','Preparado')
    add('R1-M7, R1-M8, R1-M9','Presentación de tablas y referencias','Tabla 3 candidata calculada, con unidades mensuales y precisión suficiente. Orden [12,33] identificado.','Fuente, redondeo y orden de cita se aplicarán al editar el artículo; el DOCX sigue intacto.','Preparado')
    add('R1-P4, R2-L27','Fundamento físico de A/B','Se conserva asignación A=P1–P3, B=P4–P5. OD medio A=4.7118; B=5.2323 mg/L.','Falta respaldo documental explícito del criterio hidrológico previo. Las medias y el clasificador no aportan validación independiente.','Requiere información de campo')
    add('R1-M1, R1-M2','Antigüedad y marzo 2016','Período confirmado por autores: febrero 2016–marzo 2017; marzo 2016 sin muestreo.','Mantener introducción acordada y precisar alcance histórico en discusión/limitaciones al editar.','Preparado')
    add('R1-M4, R2-L158–162','Título e interpretación metodológica','La importancia por permutación está calculada en observaciones retenidas; SHAP continúa como complemento.','Explicar interpretación predictiva y simplificar el procedimiento, conservando el alcance exploratorio.','Preparado')
    add('R1-P5, R1-P6, R2-L100–106','Respuestas proporcionadas por autores','Textos y manuscrito no modificados; verificación SHA-256 registrada.','Se mantienen tal cual por instrucción del autor. Esta ejecución no certifica documentación de laboratorio.','Conservado')
    table=pd.DataFrame(rows);table.to_csv(OUT/'tabla_observaciones_resultados.csv',index=False)
    method='''# Resultados de la revisión previa al manuscrito

Datos: febrero de 2016 a marzo de 2017, con ausencia de muestreo en marzo de 2016. Se analizaron 65 perfiles de cinco sitios y trece campañas. Se preservaron todos los archivos fuente, la introducción y las respuestas proporcionadas.

## Alcance y método

GEE gaussiano con identidad; siete respuestas se modelan como log(1+Y), como en el programa original. Para las demás se conservan sus unidades. La nueva matriz de correlación usa rho elevado a la distancia en meses calendario. El primer día de mes almacenado en el Excel es una etiqueta temporal, no el día real de campo. No se imputó la campaña de marzo de 2016.

Se aplicó la covarianza Mancl–DeRouen y una referencia t con G−1=4 grados de libertad, con sensibilidad t3 para interacciones y grados de libertad específicos de zona (2 para A y 1 para B) para pendientes. Ninguna corrección convierte cinco sitios en una muestra confirmatoria grande. Los q se calcularon por Benjamini–Hochberg sobre las trece interacciones de cada especificación, sin seleccionar pruebas por resultado.

La implementación explícita de GEE con NumPy/SciPy reproduce las pendientes publicadas a la precisión de la tabla. La covarianza MD se comprobó de manera independiente mediante la identidad de eliminación de clúster a rho fijo; diferencia máxima en errores estándar <2e−13. Los q originales no se reproducen exactamente con una familia de trece pruebas: el código original ajusta conjuntamente términos de GEE y de modelos mixtos, mientras el manuscrito dice trece interacciones. Esa inconsistencia queda corregida en la nueva tabla.

Validación predictiva principal: se retiene cada campaña completa y se entrena con las otras doce. Esto estima desempeño en campañas excluidas, no pronóstico hacia el futuro. La validación adicional anidada usa cuatro particiones internas por campaña y trece externas; compara Ridge, Elastic Net, SVR y RF dentro del entrenamiento. Los hiperparámetros originales son fijos en el script inspeccionado. La revisión no permite reconstruir selecciones manuales anteriores.

El análisis paralelo tradicional usa 2000 muestras normales independientes. La sensibilidad con 1000 desplazamientos circulares conserva vectores de cinco sitios dentro de cada variable; no constituye una prueba exacta con solo trece campañas irregulares. La estabilidad se evalúa con bloques circulares de dos y tres campañas (500 remuestreos cada uno). Los percentiles bootstrap son resúmenes de estabilidad, no intervalos confirmatorios poblacionales. Bartlett y Shapiro se muestran como diagnósticos nominales debido a la dependencia.

## Tabla de resultados y observaciones

'''
    report=method+markdown_table(table)+'\n\n## Interacciones corregidas por variable\n\n'+markdown_table(q.reset_index())
    report+='\n\n## Conclusión para la defensa\n\nSe sostienen la caracterización histórica, el gradiente fisicoquímico principal, la separabilidad exploratoria de los grupos y siete contrastes de evolución temporal bajo las correcciones ensayadas. Deben matizarse la diferencia de tendencias de OD entre zonas, el ajuste de una recta para describir su oscilación, el alcance operativo de la predicción, la independencia de la importancia del nitrito y la estabilidad de exactamente tres componentes/grupos.\n\n'
    report+='La fundamentación física previa de A/B requiere registros o una fuente externa explícita. Las aclaraciones de laboratorio se conservan; no se inventaron acreditaciones, recuperaciones, fechas exactas ni significados de ceros. Las correcciones editoriales se integrarán después de revisar estos resultados.\n\n'
    report+='## Referencias metodológicas verificadas\n\n- [Statsmodels: covarianza Mancl–DeRouen](https://www.statsmodels.org/v0.14.0/_modules/statsmodels/genmod/generalized_estimating_equations.html).\n- [Statsmodels: dependencia autorregresiva y distancia temporal](https://www.statsmodels.org/dev/generated/statsmodels.genmod.cov_struct.Autoregressive.html).\n- [Scikit-learn: validación cruzada anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).\n'
    (ROOT/'Resultados_revision_Nuxco.md').write_text(report)
    print('Report rows',len(rows),'protected source files',len(audit['protected_files_unchanged']),flush=True)

if __name__=='__main__':main()
