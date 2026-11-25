"""Additional reviewer checks. Source observations and manuscript are never edited.

Run after analysis_revision.py and gee_revision.py. All validation splits retain
complete sampling campaigns. No model or sensitivity is selected by its p-value.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json, hashlib, platform, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, t, shapiro, skew, kurtosis, spearmanr
from scipy.linalg import subspace_angles
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import ElasticNet, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
    balanced_accuracy_score, roc_auc_score, brier_score_loss, log_loss)
from analysis_revision import load_data, VARS, FEATURES, OD_FEATURES, LOG_VARS, OUT, reg_oof, metrics
from gee_revision import fit_gee, bh

ROOT = Path(__file__).parent
def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x,"item") else str(x)))

def numerical_validation(d):
    """Independent delete-cluster identity for the MD covariance at fixed rho."""
    checks=[]
    for variable in VARS:
        rr,diag=fit_gee(d,variable,calendar=True)
        tt=d.Mes_transcurrido.to_numpy();z=d.Zona_B.to_numpy()
        X=np.column_stack([np.ones(len(d)),z,tt,z*tt])
        y=d[variable].to_numpy(float)
        if variable in LOG_VARS:y=np.log1p(y)
        center=y.mean();scale=y.std();ys=(y-center)/scale
        rho=rr[0]['rho'];blocks=[]
        for site in sorted(d.Punto.unique()):
            ix=np.flatnonzero(d.Punto.to_numpy()==site)
            R=rho**np.abs(tt[ix,None]-tt[None,ix])
            blocks.append((X[ix],ys[ix],np.linalg.inv(R)))
        bmat=sum(a.T@w@a for a,v,w in blocks)
        rhs=sum(a.T@w@v for a,v,w in blocks)
        beta=np.linalg.solve(bmat,rhs)
        differences=[]
        for a,v,w in blocks:
            deleted=np.linalg.solve(bmat-a.T@w@a,rhs-a.T@w@v)
            differences.append(deleted-beta)
        jack=np.array(differences).T@np.array(differences)*scale**2
        contrasts=[np.array([0,0,1,0]),np.array([0,0,1,1]),np.array([0,0,0,1])]
        se=[float(np.sqrt(c@jack@c)) for c in contrasts]
        err=max(abs(se[i]-rr[i]['se_MD']) for i in range(3))
        reconstructed=beta*scale;reconstructed[0]+=center
        np.testing.assert_allclose(reconstructed,diag['beta'],rtol=1e-7,atol=1e-6)
        np.testing.assert_allclose(se,[r['se_MD'] for r in rr],rtol=1e-7,atol=1e-7)
        checks.append(dict(variable=variable,MD_delete_cluster_max_error=err,converged=rr[0]['converged']))
    dump('numerical_validation.json',checks)

def gee_checks(d):
    g=pd.read_csv(OUT/'gee_all_results.csv')
    for key,frame in g.groupby(['subset','working_correlation']):
        ix=frame.index[frame.contrast.eq('B_minus_A')]
        # Additional df=G-2 sensitivity. Neither t reference is exact with five sites.
        p=2*t.sf(np.abs(g.loc[ix,'beta']/g.loc[ix,'se_MD']),3)
        g.loc[ix,'p_MD_t3']=p;g.loc[ix,'q_MD_t3']=bh(p)
    g.to_csv(OUT/'gee_all_results.csv',index=False)
    diag=json.loads((OUT/'gee_diagnostics.json').read_text())
    rows=[]
    for var in VARS:
        a=diag[f'all_True_{var}'];e=np.array(a['residuals']);pred=np.array(a['fitted'])
        original=d[var].to_numpy(float)
        fitted_original=np.expm1(pred) if var in LOG_VARS else pred
        for label,mask in [('all',np.ones(len(d),bool)),('A',d.Zona_B.eq(0).to_numpy()),('B',d.Zona_B.eq(1).to_numpy())]:
            met=metrics(original[mask],fitted_original[mask])
            rows.append(dict(variable=var,zone=label,**met,
                residual_skew=float(skew(e[mask])),residual_excess_kurtosis=float(kurtosis(e[mask])),
                Shapiro_p_nominal_only=float(shapiro(e[mask]).pvalue),
                abs_residual_fitted_rho=float(spearmanr(np.abs(e[mask]),pred[mask]).statistic)))
        pd.DataFrame({'ID':d.ID_muestra,'period':d.Periodo,'site':d.Punto,
            'zone':np.where(d.Zona_B,'B','A'),'fitted_model_scale':pred,'residual_model_scale':e,
            'observed_original_scale':original,'fitted_original_scale':fitted_original}).to_csv(OUT/f'residuals_{var}.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'residual_diagnostics.csv',index=False)
    # Site-deletion sensitivity: coefficients only, as the independent sample is even smaller.
    site_rows=[]
    for site in sorted(d.Punto.unique()):
        ds=d.loc[d.Punto.ne(site)].reset_index(drop=True)
        for var in ['OD_mg_L','DBO5_mg_L','NO3_mg_L']:
            _,dd=fit_gee(ds,var,calendar=True,compute_covariance=False)
            beta=dd['beta']
            for contrast,value in [('A',beta[2]),('B',beta[2]+beta[3]),('B_minus_A',beta[3])]:
                site_rows.append(dict(removed_site=int(site),variable=var,contrast=contrast,beta=value,
                                      rho=dd['rho'],converged=dd['converged']))
    pd.DataFrame(site_rows).to_csv(OUT/'gee_leave_site_out.csv',index=False)
    numerical_validation(d)
    print('GEE validation, diagnostics and site sensitivity complete',flush=True)

def collinearity(d):
    results={}
    for name,cols in [('all13',VARS),('PCA11',FEATURES),('OD10',OD_FEATURES)]:
        X=StandardScaler().fit_transform(d[cols]);R=np.corrcoef(X,rowvar=False)
        inv=np.linalg.inv(R);partial=-inv/np.sqrt(np.outer(inv.diagonal(),inv.diagonal()))
        np.fill_diagonal(partial,0);R0=R.copy();np.fill_diagonal(R0,0)
        kmo=(R0**2).sum()/((R0**2).sum()+(partial**2).sum())
        kmo_v=(R0**2).sum(axis=0)/((R0**2).sum(axis=0)+(partial**2).sum(axis=0))
        n,p=X.shape;stat=-(n-1-(2*p+5)/6)*np.linalg.slogdet(R)[1]
        results[name]=dict(VIF=dict(zip(cols,inv.diagonal())),KMO=kmo,KMO_variables=dict(zip(cols,kmo_v)),
            Bartlett_chi2=stat,Bartlett_df=p*(p-1)//2,Bartlett_p_nominal_only=float(chi2.sf(stat,p*(p-1)//2)))
    dump('collinearity.json',results)
    d[FEATURES].corr().to_csv(OUT/'correlations_pearson.csv')
    d[FEATURES].corr(method='spearman').to_csv(OUT/'correlations_spearman.csv')

def stability(d):
    X=d[FEATURES].to_numpy();Z=StandardScaler().fit_transform(X)
    p0=PCA().fit(Z);ref=AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(Z)
    periods=sorted(d.Periodo.unique());blocks=[np.flatnonzero(d.Periodo.eq(p)) for p in periods]
    rng=np.random.default_rng(20261113);rows=[]
    # Moving blocks preserve simultaneous five-site observations and local temporal order.
    for length in [2,3]:
        for rep in range(500):
            starts=rng.integers(len(periods),size=int(np.ceil(len(periods)/length)))
            times=np.concatenate([(np.arange(length)+s)%len(periods) for s in starts])[:len(periods)]
            ix=np.concatenate([blocks[i] for i in times]);unique=np.unique(ix)
            zb=StandardScaler().fit_transform(X[ix]);p=PCA().fit(zb)
            lab=AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(zb)
            # Duplicated observations share a value; use modal membership if a tie is split.
            mapped=np.array([np.bincount(lab[ix==j]).argmax() for j in unique])
            ari=adjusted_rand_score(ref[unique],mapped)
            jacc=[]
            for c in np.unique(ref):
                a=ref[unique]==c
                if not a.any():jacc.append(np.nan);continue
                jacc.append(max(np.sum(a&(mapped==k))/np.sum(a|(mapped==k)) for k in np.unique(mapped)))
            cos=np.abs(np.diag(p.components_[:3]@p0.components_[:3].T))
            angles=np.rad2deg(subspace_angles(p.components_[:3].T,p0.components_[:3].T))
            rows.append(dict(block_length=length,repeat=rep,n_unique=len(unique),ARI_Ward_inbag=ari,
                min_cluster_Jaccard=float(np.nanmin(jacc)),PC1_cos=cos[0],PC2_cos=cos[1],PC3_cos=cos[2],
                max_subspace_angle_deg=angles.max(),variance3=p.explained_variance_ratio_[:3].sum()))
        print('Ward/PCA bootstrap block',length,'complete',flush=True)
    b=pd.DataFrame(rows);b.to_csv(OUT/'bootstrap_Ward_PCA.csv',index=False)
    summary={str(length):f.drop(columns=['repeat','block_length']).quantile([.025,.5,.975]).to_dict() for length,f in b.groupby('block_length')}
    # Dependence-aware descriptive sensitivity to the iid Gaussian parallel analysis.
    cube=np.stack([X[ix] for ix in blocks]);null=[]
    for rep in range(1000):
        shifted=cube.copy()
        for j in range(X.shape[1]):shifted[:,:,j]=np.roll(cube[:,:,j],int(rng.integers(len(periods))),axis=0)
        null.append(PCA().fit(StandardScaler().fit_transform(shifted.reshape(X.shape))).explained_variance_)
    threshold=np.quantile(null,.95,axis=0)
    summary['circular_shift_parallel']=dict(repetitions=1000,threshold95=threshold.tolist(),
        retained=int(np.sum(p0.explained_variance_>threshold)),
        note='Descriptive sensitivity, circular campaign shifts preserve five-site profiles and temporal sequence within each variable; not an exact null for irregular dates.')
    dump('stability_extended.json',summary)
    pd.DataFrame(p0.components_.T,index=FEATURES,columns=[f'PC{i+1}' for i in range(len(FEATURES))]).to_csv(OUT/'PCA_loadings.csv')
    scores=pd.DataFrame(p0.transform(Z)[:,:3],columns=['PC1','PC2','PC3'])
    scores.insert(0,'ID',d.ID_muestra);scores['zone']=np.where(d.Zona_B,'B','A');scores.to_csv(OUT/'PCA_scores.csv',index=False)

def nested_ml(d):
    X=d[OD_FEATURES];y=d.OD_mg_L.to_numpy();groups=d.Periodo.to_numpy()
    # Model-family and hyperparameter selection both occur in outer training only.
    grids=[
        {'model':[Ridge()],'model__alpha':[.1,1.,10.,100.]},
        {'model':[ElasticNet(max_iter=20000,random_state=42)],'model__alpha':[.01,.05,.1,.5],'model__l1_ratio':[.1,.5,.9]},
        {'model':[SVR(gamma='scale')],'model__C':[.1,1.,10.],'model__epsilon':[.05,.1]},
        {'model':[RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=1)],'model__max_depth':[3,4],'model__min_samples_leaf':[3,5]}
    ]
    pred=np.full(len(d),np.nan);base=pred.copy();selections=[];prob=np.full(len(d),np.nan)
    target=(1-d.Zona_B).to_numpy()
    for number,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups),1):
        inner=list(GroupKFold(n_splits=4).split(X.iloc[tr],y[tr],groups[tr]))
        # Confirm all nested partitions are disjoint by campaign.
        for a,b in inner:assert set(groups[tr][a]).isdisjoint(groups[tr][b])
        search=GridSearchCV(Pipeline([('scale',StandardScaler()),('model',Ridge())]),grids,
                            scoring='neg_mean_squared_error',cv=inner,n_jobs=1,error_score='raise')
        search.fit(X.iloc[tr],y[tr]);pred[te]=search.predict(X.iloc[te]);base[te]=y[tr].mean()
        clf=GridSearchCV(Pipeline([('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',max_iter=5000,random_state=42))]),
                         {'model__C':[.01,.1,1.,10.,100.]},scoring='neg_log_loss',cv=inner,n_jobs=1,error_score='raise')
        clf.fit(d[FEATURES].iloc[tr],target[tr]);prob[te]=clf.predict_proba(d[FEATURES].iloc[te])[:,1]
        selections.append(dict(period=str(groups[te][0]),regression_params=str(search.best_params_),
             inner_RMSE=float(np.sqrt(-search.best_score_)),classifier_C=clf.best_params_['model__C']))
        print('Nested outer campaign',number,'of 13',flush=True)
    pd.DataFrame({'ID':d.ID_muestra,'period':groups,'observed':y,'predicted':pred,'baseline':base,'class_true':target,'class_probability':prob}).to_csv(OUT/'nested_oof.csv',index=False)
    dump('nested_validation.json',dict(regression=metrics(y,pred),baseline=metrics(y,base),
         classification=dict(balanced_accuracy=float(balanced_accuracy_score(target,prob>=.5)),
            AUC=float(roc_auc_score(target,prob)),Brier=float(brier_score_loss(target,prob)),LogLoss=float(log_loss(target,prob))),
         selections=selections,inner_folds=4,outer_folds=13))

def zero_sensitivity(d):
    model=make_pipeline(StandardScaler(),ElasticNet(alpha=.05,l1_ratio=.5,max_iter=20000,random_state=42))
    scenarios={
        'unchanged':d.copy(),
        'exclude_NO2_zero':d.loc[d.NO2_mg_L.ne(0)].reset_index(drop=True),
        'exclude_any_zero_NO2_SS':d.loc[d.NO2_mg_L.ne(0)&d.SS_mL_L.ne(0)].reset_index(drop=True)}
    for label,value in [('LD_half',.000974/2),('LC_half',.01/2)]:
        ds=d.copy();ds.loc[ds.NO2_mg_L.eq(0),'NO2_mg_L']=value;scenarios[label]=ds
    rows=[]
    for label,ds in scenarios.items():
        met,p,base,_=reg_oof(ds,model)
        rows.append(dict(scenario=label,**met,baseline_RMSE=metrics(ds.OD_mg_L,base)['RMSE']))
    dump('zero_sensitivity.json',dict(results=rows,note='Hypothetical sensitivity only. No claim that zeros are non-detects; no original value changed. LD/LC use reported units pending confirmation of N-versus-ion basis.'))

def main():
    d=load_data();assert d.Fecha.min().year==2016 and d.Fecha.max().year==2017
    gee_checks(d);collinearity(d);zero_sensitivity(d);stability(d);nested_ml(d)
    import scipy,sklearn,matplotlib
    dump('environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,
        note='Gaussian identity GEE and MD sandwich computed explicitly with NumPy/SciPy; checked by independent fixed-rho delete-cluster identity.'))
    print('ALL EXTENDED CHECKS COMPLETED',flush=True)

if __name__=='__main__':main()
