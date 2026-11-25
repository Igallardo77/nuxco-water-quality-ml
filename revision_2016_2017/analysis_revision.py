"""Revised analysis of historical Nuxco inputs; source measurements are never edited."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (silhouette_score, adjusted_rand_score, r2_score,
    mean_squared_error, mean_absolute_error, balanced_accuracy_score, roc_auc_score,
    brier_score_loss, log_loss, confusion_matrix)

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SOURCE = ROOT.parent / "data/Base_Datos_Calidad_Agua_Integrada.xlsx"
VARS = ["Temperatura_C", "pH", "CE_mS_cm", "SDT_mg_L", "Salinidad_UPS",
        "OD_mg_L", "DBO5_mg_L", "DQO_mg_L", "SST_mg_L", "SS_mL_L",
        "NH3_N_mg_L", "NO2_mg_L", "NO3_mg_L"]
FEATURES = [v for v in VARS if v not in ("SDT_mg_L", "Salinidad_UPS")]
OD_FEATURES = [v for v in FEATURES if v != "OD_mg_L"]
LOG_VARS = {"DBO5_mg_L", "DQO_mg_L", "SST_mg_L", "SS_mL_L", "NH3_N_mg_L", "NO2_mg_L", "NO3_mg_L"}

def load_data():
    d = pd.read_excel(SOURCE, sheet_name="01_Dataset_ML")
    d["Fecha_original"] = d.Fecha
    d["ID_original"] = d.ID_muestra
    # Final author-confirmed period: February 2016 to March 2017. Dates in the workbook are month labels, not verified sampling days.
    d["Fecha"] = pd.to_datetime(d.Fecha)
    d["Anio"] = d.Fecha.dt.year
    d["Periodo"] = d.Fecha.dt.strftime("%Y-%m")
    d["ID_muestra"] = d.Fecha.dt.strftime("%Y%m") + "_P" + d.Punto.astype(str)
    d["Mes_transcurrido"] = ((d.Fecha.dt.year - d.Fecha.dt.year.min()) * 12 + d.Fecha.dt.month - d.Fecha.dt.month.iloc[0]).astype(float)
    d["Zona_B"] = (d.Punto >= 4).astype(int)
    return d

def metrics(y, p):
    return dict(n=len(y), MAE=float(mean_absolute_error(y,p)),
        RMSE=float(mean_squared_error(y,p)**.5), R2=float(r2_score(y,p)),
        Bias=float(np.mean(p-y)))

def reg_oof(d, model, features=OD_FEATURES, group="Periodo"):
    X=d[features];y=d.OD_mg_L.to_numpy();p=np.full(len(d),np.nan);base=p.copy(); folds=[]
    for tr,te in LeaveOneGroupOut().split(X,y,d[group]):
        fitted=clone(model).fit(X.iloc[tr],y[tr])
        p[te]=fitted.predict(X.iloc[te]);base[te]=y[tr].mean()
        folds.append((fitted,te))
    return metrics(y,p),p,base,folds

def main():
    d=load_data();result={}
    result["qc"]={"n":len(d),"sites":int(d.Punto.nunique()),"campaigns":int(d.Periodo.nunique()),
        "measurements":int(d[VARS].size),"missing":int(d[VARS].isna().sum().sum()),
        "duplicate_ids":int(d.ID_muestra.duplicated().sum()),
        "duplicate_site_period":int(d.duplicated(["Punto","Periodo"]).sum()),
        "zero_counts":d[VARS].eq(0).sum()[lambda z:z>0].to_dict(),
        "pH_arithmetic_mean":float(d.pH.mean()),"pH_activity_mean":float(-np.log10(np.mean(10.**(-d.pH)))),
        "pH_median":float(d.pH.median()),"pH_q25":float(d.pH.quantile(.25)),"pH_q75":float(d.pH.quantile(.75))}
    d.groupby(["Periodo","Zona_B"])[["OD_mg_L","DBO5_mg_L","NO3_mg_L"]].mean().to_csv(OUT/"monthly_zone_means.csv")
    print("QC",result["qc"],flush=True)
    Z=StandardScaler().fit_transform(d[FEATURES]);pca=PCA().fit(Z)
    rng=np.random.default_rng(42)
    null=np.array([PCA().fit(StandardScaler().fit_transform(rng.normal(size=Z.shape))).explained_variance_ for _ in range(2000)])
    thresholds=np.quantile(null,.95,axis=0)
    R=np.corrcoef(Z,rowvar=False);invR=np.linalg.inv(R)
    partial=-invR/np.sqrt(np.outer(np.diag(invR),np.diag(invR)))
    np.fill_diagonal(R,0);np.fill_diagonal(partial,0)
    kmo=float((R**2).sum()/((R**2).sum()+(partial**2).sum()))
    result["pca"]={"variance_ratio":pca.explained_variance_ratio_.tolist(),"eigenvalues":pca.explained_variance_.tolist(),
        "parallel95":thresholds.tolist(),"parallel_retain":int((pca.explained_variance_>thresholds).sum()),"KMO":kmo}
    cluster=[]
    for k in (2,3,4):
        km=KMeans(n_clusters=k,n_init=50,random_state=42).fit(Z)
        cluster.append({"k":k,"silhouette":float(silhouette_score(Z,km.labels_)),
            "sizes":np.bincount(km.labels_).tolist(),"ARI_Ward":float(adjusted_rand_score(km.labels_,AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(Z)))})
    result["clusters"]=cluster
    print("PCA",result["pca"],"CLUSTERS",cluster,flush=True)
    z=RobustScaler().fit_transform(d[FEATURES])
    f=IsolationForest(n_estimators=500,contamination=.1,random_state=42).fit_predict(z)==-1
    l=LocalOutlierFactor(n_neighbors=15,contamination=.1).fit_predict(z)==-1
    m=MinCovDet(random_state=42,support_fraction=.8).fit(z).mahalanobis(z)>chi2.ppf(.9,len(FEATURES))
    consensus=f.astype(int)+l.astype(int)+m.astype(int)
    flags=pd.DataFrame({"ID":d.ID_muestra,"IF":f,"LOF":l,"MCD":m,"votes":consensus})
    flags.to_csv(OUT/"anomaly_flags.csv",index=False)
    result["anomalies"]=flags.loc[consensus>=2].to_dict("records")
    print("ANOMALIES",result["anomalies"],flush=True)
    regs={"ElasticNet":make_pipeline(StandardScaler(),ElasticNet(alpha=.05,l1_ratio=.5,max_iter=20000,random_state=42)),
        "Ridge":make_pipeline(StandardScaler(),Ridge(alpha=1)),
        "SVR":make_pipeline(StandardScaler(),SVR(C=10,epsilon=.1,gamma="scale")),
        "RF":RandomForestRegressor(n_estimators=300,max_depth=4,min_samples_leaf=3,random_state=42)}
    rows=[]
    for group in ("Periodo","Punto"):
        for name,model in regs.items():
            met,p,b,folds=reg_oof(d,model,group=group)
            rows.append(dict(group=group,model=name,**met))
            pd.DataFrame({"ID":d.ID_muestra,"period":d.Periodo,"site":d.Punto,"real":d.OD_mg_L,"pred":p,"baseline":b}).to_csv(OUT/f"oof_{group}_{name}.csv",index=False)
    result["regression"]=rows
    print("REGRESSION",rows,flush=True)
    clf={"Logistic":make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight="balanced",max_iter=5000,random_state=42)),
        "SVM":make_pipeline(StandardScaler(),SVC(kernel="linear",C=1,probability=True,class_weight="balanced",random_state=42)),
        "RF":RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=3,class_weight="balanced",random_state=42)}
    cl=[];X=d[FEATURES];y=(d.Punto<=3).astype(int).to_numpy()
    for group in ("Periodo","Punto"):
        for name,model in clf.items():
            prob=np.full(len(d),np.nan)
            for tr,te in LeaveOneGroupOut().split(X,y,d[group]):
                fitted=clone(model).fit(X.iloc[tr],y[tr]);prob[te]=fitted.predict_proba(X.iloc[te])[:,1]
            row=dict(group=group,model=name,balanced_accuracy=float(balanced_accuracy_score(y,prob>=.5)),
                AUC=float(roc_auc_score(y,prob)),Brier=float(brier_score_loss(y,prob)),LogLoss=float(log_loss(y,prob)),confusion=confusion_matrix(y,prob>=.5).tolist())
            cl.append(row)
            pd.DataFrame({"ID":d.ID_muestra,"real":y,"prob":prob,"period":d.Periodo}).to_csv(OUT/f"clf_{group}_{name}.csv",index=False)
    result["classification"]=cl;print("CLASSIFICATION",cl,flush=True)
    subsets={"all":d,"without_consensus":d.loc[consensus<2].reset_index(drop=True),
        "without_Oct_Nov_2016":d.loc[~d.Periodo.isin(["2016-10","2016-11"])].reset_index(drop=True)}
    sens=[]
    for subset,ds in subsets.items():
        for feat in (OD_FEATURES,[v for v in OD_FEATURES if v!="NO2_mg_L"]):
            met,p,b,folds=reg_oof(ds,regs["ElasticNet"],features=feat)
            base=metrics(ds.OD_mg_L.to_numpy(),b)
            sens.append(dict(subset=subset,nitrite=("NO2_mg_L" in feat),**met,baseline_RMSE=base["RMSE"],baseline_R2=base["R2"]))
            if subset=="all" and "NO2_mg_L" in feat:
                y0=ds.OD_mg_L.to_numpy()
                for mask_name,mask in [("OD_le_4",y0<=4),("OD_gt_4",y0>4)]:
                    result[mask_name]=dict(**metrics(y0[mask],p[mask]),baseline=metrics(y0[mask],b[mask]))
    result["regression_sensitivity"]=sens;print("REGRESSION_SENSITIVITY",sens,flush=True)
    (OUT/"analysis_results.json").write_text(json.dumps(result,indent=2,default=lambda a:a.item() if hasattr(a,"item") else str(a)))
    d.to_csv(OUT/"analysis_data_2016_2017.csv",index=False)

if __name__=="__main__":main()
