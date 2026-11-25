"""Diagnostics, dependence-aware stability and detector-parameter sensitivity."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import json
import numpy as np,pandas as pd
from scipy.stats import chi2,spearmanr
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import adjusted_rand_score,silhouette_score,mean_squared_error
from analysis_revision import load_data,FEATURES,OD_FEATURES,OUT,reg_oof

def main():
    d=load_data();X=d[FEATURES].to_numpy();Z=StandardScaler().fit_transform(X)
    R=np.corrcoef(X,rowvar=False);ri=np.linalg.inv(R);sign,ld=np.linalg.slogdet(R)
    result={"VIF":dict(zip(FEATURES,np.diag(ri).tolist())),
        "Bartlett_nominal_independence_only":{"chi2":float(-(len(d)-1-(2*len(FEATURES)+5)/6)*ld),"df":55},
        "correlation_Pearson_OD":d[FEATURES].corr().OD_mg_L.to_dict(),
        "correlation_Spearman_OD":d[FEATURES].corr(method="spearman").OD_mg_L.to_dict()}
    pca=PCA().fit(Z);reference=pca.components_[:3]
    km=KMeans(n_clusters=3,n_init=100,random_state=42).fit(Z)
    periods=sorted(d.Periodo.unique());blocks=[np.flatnonzero(d.Periodo.to_numpy()==p) for p in periods]
    rng=np.random.default_rng(20261108);stability=[]
    # Circular moving blocks of two successive campaigns; five sites retained together.
    for repeat in range(500):
        starts=rng.integers(0,len(periods),size=7)
        take=np.concatenate([np.array([s,(s+1)%len(periods)]) for s in starts])[:len(periods)]
        ix=np.concatenate([blocks[j] for j in take])
        scaler=StandardScaler().fit(X[ix]);zb=scaler.transform(X[ix]);za=scaler.transform(X)
        p=PCA().fit(zb)
        cos=np.abs(np.sum(p.components_[:3]*reference,axis=1))
        scores=[];best_k=None;best_s=-np.inf
        for k in (2,3,4):
            fitted=KMeans(n_clusters=k,n_init=10,random_state=42).fit(zb)
            s=silhouette_score(zb,fitted.labels_);scores.append(s)
            if s>best_s:best_s=s;best_k=k
            if k==3:ari=adjusted_rand_score(km.labels_,fitted.predict(za))
        stability.append(dict(repeat=repeat,best_k=best_k,ARI_k3=float(ari),
            cosine_PC1=float(cos[0]),cosine_PC2=float(cos[1]),cosine_PC3=float(cos[2]),
            variance_PC1=float(p.explained_variance_ratio_[0]),variance_PC2=float(p.explained_variance_ratio_[1]),variance_PC3=float(p.explained_variance_ratio_[2])))
        if repeat%100==0:print("BOOTSTRAP",repeat,flush=True)
    boot=pd.DataFrame(stability);boot.to_csv(OUT/"bootstrap_temporal_blocks.csv",index=False)
    result["bootstrap"]={"repetitions":500,"block_campaigns":2,
        "best_k_counts":boot.best_k.value_counts().sort_index().to_dict(),
        "quantiles":boot.drop(columns=["repeat","best_k"]).quantile([.025,.5,.975]).to_dict()}
    z=RobustScaler().fit_transform(d[FEATURES]);forest=IsolationForest(n_estimators=500,contamination=.1,random_state=42).fit(z)
    fs=forest.score_samples(z);md=MinCovDet(random_state=42,support_fraction=.8).fit(z).mahalanobis(z)
    detector=[]
    core={"201610_P1","201610_P2","201611_P1","201611_P2"}
    for contamination in [.05,.10,.15]:
        fi=fs<np.percentile(fs,contamination*100)
        for neighbors in [10,15,20]:
            lof=LocalOutlierFactor(n_neighbors=neighbors,contamination=contamination).fit_predict(z)==-1
            for quantile in [.90,.95]:
                ma=md>chi2.ppf(quantile,len(FEATURES))
                vote=fi.astype(int)+lof.astype(int)+ma.astype(int)
                ids=d.loc[vote>=2,"ID_muestra"].tolist()
                detector.append(dict(contamination=contamination,neighbors=neighbors,MCD_quantile=quantile,n=len(ids),all_four_events_flagged=core.issubset(ids),ids=ids))
    result["detector_sensitivity"]=detector
    flags=pd.read_csv(OUT/"anomaly_flags.csv");subset=d.loc[~d.ID_muestra.isin(flags.loc[flags.votes>=2,"ID"])].reset_index(drop=True)
    model=make_pipeline(StandardScaler(),ElasticNet(alpha=.05,l1_ratio=.5,max_iter=20000,random_state=42))
    importance=[]
    for name,ds in [("all",d),("without_consensus",subset)]:
        met,p,base,folds=reg_oof(ds,model)
        rng=np.random.default_rng(42);xx=ds[OD_FEATURES];y=ds.OD_mg_L.to_numpy()
        for feature in OD_FEATURES:
            values=[]
            for rep in range(30):
                pp=p.copy()
                for fitted,ix in folds:
                    test=xx.iloc[ix].copy();test[feature]=rng.permutation(test[feature].to_numpy())
                    pp[ix]=fitted.predict(test)
                values.append(mean_squared_error(y,pp)**.5-met["RMSE"])
            importance.append(dict(subset=name,variable=feature,DeltaRMSE=float(np.mean(values)),SD_permutation=float(np.std(values,ddof=1))))
        print("IMPORTANCE",name,flush=True)
    result["permutation_importance"]=importance
    p=pd.read_csv(OUT/"oof_Periodo_ElasticNet.csv")
    p["squared_error"]=(p.pred-p.real)**2;p["absolute_error"]=abs(p.pred-p.real)
    fold=p.groupby("period").agg(n=("real","size"),MSE=("squared_error","mean"),MAE=("absolute_error","mean"))
    fold["RMSE"]=np.sqrt(fold.MSE);fold.to_csv(OUT/"OD_fold_errors.csv")
    result["fold_error_extremes"]={"min_RMSE":float(fold.RMSE.min()),"max_RMSE":float(fold.RMSE.max()),"worst_period":str(fold.RMSE.idxmax())}
    cal={}
    for name in ["Logistic","SVM","RF"]:
        c=pd.read_csv(OUT/f"clf_Periodo_{name}.csv")
        c["bin"]=pd.cut(c.prob,bins=np.linspace(0,1,6),include_lowest=True)
        tab=c.groupby("bin",observed=True).agg(n=("real","size"),predicted=("prob","mean"),observed=("real","mean"))
        cal[name]=[{"bin":str(i),**row} for i,row in tab.to_dict("index").items()]
    result["calibration_bins"]=cal
    (OUT/"diagnostic_results.json").write_text(json.dumps(result,indent=2,default=lambda a:a.item() if hasattr(a,"item") else str(a)))
    print("DIAGNOSTICS",json.dumps({k:v for k,v in result.items() if k not in ["detector_sensitivity","calibration_bins"]},default=str),flush=True)

if __name__=="__main__":main()
