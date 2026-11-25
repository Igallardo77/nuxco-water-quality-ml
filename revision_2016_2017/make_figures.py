"""Readable scientific figures and final numerical summaries; no manuscript edits."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import probplot,t
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from analysis_revision import load_data,VARS,FEATURES,LOG_VARS,OUT

ROOT=Path(__file__).parent;FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
LABELS={'Temperatura_C':'Temperature (°C)','pH':'pH','CE_mS_cm':'EC (mS/cm)',
 'SDT_mg_L':'TDS (mg/L)','Salinidad_UPS':'Salinity (PSU)','OD_mg_L':'DO (mg/L)',
 'DBO5_mg_L':'BOD5 (mg/L)','DQO_mg_L':'COD (mg/L)','SST_mg_L':'TSS (mg/L)',
 'SS_mL_L':'Settleable solids (mL/L)','NH3_N_mg_L':'NH3–N (mg/L)',
 'NO2_mg_L':'Nitrite (reported mg/L)','NO3_mg_L':'Nitrate (reported mg/L)'}
SHORT={k:v.split(' (')[0] for k,v in LABELS.items()}
COL={'A':'#A63D40','B':'#245F88'}
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'legend.fontsize':10,
 'savefig.dpi':250,'figure.constrained_layout.use':True,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=250);fig.savefig(FIG/(name+'.svg'));plt.close(fig)
def read(name):return json.loads((OUT/name).read_text())

def main(only=None):
    d=load_data();a=read('analysis_results.json');diag=read('gee_diagnostics.json');dr=read('diagnostic_results.json')
    g=pd.read_csv(OUT/'gee_all_results.csv')
    # Additional small-sample reference sensitivity for within-zone slopes.
    for zone,df_ref in [('A',2),('B',1)]:
        ix=g.index[g.contrast.eq(zone)]
        stat=np.abs(g.loc[ix,'beta']/g.loc[ix,'se_MD'])
        g.loc[ix,'df_within_zone_sensitivity']=df_ref
        g.loc[ix,'p_MD_zone_df_sensitivity']=2*t.sf(stat,df_ref)
        c=t.ppf(.975,df_ref)
        g.loc[ix,'CI95_zone_df_low']=g.loc[ix,'beta']-c*g.loc[ix,'se_MD']
        g.loc[ix,'CI95_zone_df_high']=g.loc[ix,'beta']+c*g.loc[ix,'se_MD']
    g.to_csv(OUT/'gee_all_results.csv',index=False)
    base=g.query("subset=='all' and working_correlation=='calendar_month'")
    table=[]
    for var in VARS:
        f=base.loc[base.variable.eq(var)].set_index('contrast');row={'variable':var,
          'scale':'log(1+Y) per month' if var in LOG_VARS else 'original unit per month'}
        for c in ['A','B','B_minus_A']:
            for k in ['beta','se_MD','CI95_MD_t_low','CI95_MD_t_high','p_t_MD']:
                row[c+'_'+k]=f.loc[c,k]
        row['q_interaction_MD_t4']=f.loc['B_minus_A','q_p_t_MD']
        row['q_interaction_MD_t3']=f.loc['B_minus_A','q_MD_t3'];table.append(row)
    pd.DataFrame(table).to_csv(OUT/'Table3_corrected_candidates.csv',index=False)
    periods=d[['Periodo','Mes_transcurrido']].drop_duplicates().sort_values('Mes_transcurrido')
    for var in VARS:
        if only is not None and var != only:continue
        x=pd.read_csv(OUT/f'residuals_{var}.csv');fig,axs=plt.subplots(1,2,figsize=(11,4.5))
        for zone,f in x.groupby('zone'):
            axs[0].scatter(f.fitted_model_scale,f.residual_model_scale,label='Zone '+zone,color=COL[zone],s=35,alpha=.8)
        axs[0].axhline(0,color='#555555',lw=1);axs[0].set(xlabel='Fitted (model scale)',ylabel='Residual',title='Residuals versus fitted');axs[0].legend()
        (osm,osr),(slope,intercept,r)=probplot(x.residual_model_scale,dist='norm')
        axs[1].scatter(osm,osr,s=30,color='#245F88');axs[1].plot(osm,slope*osm+intercept,color='#A63D40')
        axs[1].set(xlabel='Normal theoretical quantile',ylabel='Ordered residual',title='Normal Q–Q (descriptive)')
        fig.suptitle(SHORT[var]+(' — log(1+Y)' if var in LOG_VARS else ''))
        save(fig,'diagnostic_'+var)
        fig,ax=plt.subplots(figsize=(10,5))
        dd=diag[f'all_True_{var}'];b=np.array(dd['beta']);tx=np.linspace(0,13,150)
        for zone,z in [('A',0),('B',1)]:
            f=d.loc[d.Zona_B.eq(z)];monthly=f.groupby('Mes_transcurrido')[var].mean()
            ax.scatter(f.Mes_transcurrido,f[var],color=COL[zone],s=22,alpha=.4)
            ax.plot(monthly.index,monthly.values,color=COL[zone],marker='o',lw=1,label='Zone '+zone+' campaign mean')
            fit=b[0]+b[1]*z+(b[2]+b[3]*z)*tx
            if var in LOG_VARS:fit=np.expm1(fit)
            ax.plot(tx,fit,color=COL[zone],ls='--',lw=2,label='Zone '+zone+' GEE trend')
        ax.set(xticks=periods.Mes_transcurrido,xlabel='Sampling campaign (calendar spacing)',ylabel=LABELS[var],title=SHORT[var]+' across 13 campaigns')
        ax.set_xticklabels(periods.Periodo,rotation=55,ha='right');ax.legend(ncol=2)
        if var=='OD_mg_L':
            ax.annotate('Apr 2016: both zone means = 5.89 mg/L',xy=(2,5.89),xytext=(3.3,6.05),
                        fontsize=9,arrowprops={'arrowstyle':'->','color':'#555555'})
        save(fig,'trajectory_'+var)
    if only is not None:return
    # Scree and explicit parallel-analysis comparisons.
    pc=a['pca'];shift=read('stability_extended.json')['circular_shift_parallel']
    fig,ax=plt.subplots(figsize=(8,5));xx=np.arange(1,12)
    ax.plot(xx,pc['eigenvalues'],'o-',label='Observed eigenvalue')
    ax.plot(xx,pc['parallel95'],'s--',label='95th percentile: iid normal parallel analysis')
    ax.plot(xx,shift['threshold95'],'^:',label='95th percentile: circular campaign-shift sensitivity')
    ax.set(xticks=xx,xlabel='Principal component',ylabel='Eigenvalue',title='Component retention diagnostics');ax.legend()
    save(fig,'PCA_scree_parallel')
    scores=pd.read_csv(OUT/'PCA_scores.csv');loads=pd.read_csv(OUT/'PCA_loadings.csv',index_col=0)
    for xpc,ypc in [('PC1','PC2'),('PC1','PC3'),('PC2','PC3')]:
        fig,ax=plt.subplots(figsize=(8,6))
        for zone,f in scores.groupby('zone'):ax.scatter(f[xpc],f[ypc],label='Zone '+zone,color=COL[zone],s=45,alpha=.8)
        ax.axhline(0,color='#bbbbbb',lw=.8);ax.axvline(0,color='#bbbbbb',lw=.8)
        ax.set(xlabel=f'{xpc} ({100*pc["variance_ratio"][int(xpc[-1])-1]:.1f}%)',ylabel=f'{ypc} ({100*pc["variance_ratio"][int(ypc[-1])-1]:.1f}%)',title='Physicochemical profiles');ax.legend()
        save(fig,'PCA_scores_'+xpc+'_'+ypc)
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(loads.iloc[:,:3],cmap='RdBu_r',vmin=-.65,vmax=.65,aspect='auto')
    ax.set(xticks=range(3),xticklabels=['PC1','PC2','PC3'],yticks=range(11),yticklabels=[SHORT[v] for v in FEATURES],title='PCA eigenvector coefficients')
    for i in range(11):
        for j in range(3):ax.text(j,i,f'{loads.iloc[i,j]:.3f}',ha='center',va='center',color='black',fontsize=10)
    fig.colorbar(im,ax=ax,label='Coefficient');save(fig,'PCA_loadings')
    Z=StandardScaler().fit_transform(d[FEATURES]);cl=AgglomerativeClustering(n_clusters=3,linkage='ward').fit_predict(Z)
    membership=d[['ID_muestra','Periodo','Punto','Zona_exploratoria']].copy();membership['Ward_cluster']=cl+1;membership.to_csv(OUT/'Ward_membership.csv',index=False)
    fig,ax=plt.subplots(figsize=(19,6));dendrogram(linkage(Z,method='ward'),labels=d.ID_muestra.to_list(),leaf_rotation=90,leaf_font_size=9,ax=ax)
    ax.set(ylabel='Ward linkage distance',title='Ward hierarchical clustering');save(fig,'Ward_dendrogram')
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot([r['k'] for r in a['clusters']],[r['silhouette'] for r in a['clusters']],'o-',color='#245F88')
    ax.set(xticks=[2,3,4],xlabel='Number of clusters',ylabel='Mean silhouette',title='Candidate cluster solutions');save(fig,'cluster_silhouette')
    # Calibration bins are descriptive: 65 dependent observations, not iid binomial trials.
    fig,ax=plt.subplots(figsize=(7,6));ax.plot([0,1],[0,1],'--',color='#888888',label='Perfect calibration')
    for name,bins in dr['calibration_bins'].items():
        ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=name)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean predicted probability of Zone A',ylabel='Observed proportion of Zone A',title='Out-of-fold classifier calibration');ax.legend();save(fig,'classifier_calibration')
    fig,ax=plt.subplots(figsize=(7,6))
    for name in ['Logistic','SVM','RF']:
        f=pd.read_csv(OUT/f'clf_Periodo_{name}.csv');fp,tp,_=roc_curve(f.real,f.prob)
        ax.plot(fp,tp,label=f'{name} (AUC {roc_auc_score(f.real,f.prob):.3f})')
    ax.plot([0,1],[0,1],'--',color='#888888');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Out-of-fold ROC curves');ax.legend();save(fig,'classifier_ROC')
    o=pd.read_csv(OUT/'oof_Periodo_ElasticNet.csv');nested=pd.read_csv(OUT/'nested_oof.csv')
    for name,pred in [('ElasticNet_fixed',o.pred),('nested_model_selection',nested.predicted)]:
        fig,ax=plt.subplots(figsize=(7,6));mask=o.real.le(4)
        for m,label,color in [(~mask,'Observed DO > 4 mg/L','#245F88'),(mask,'Observed DO ≤ 4 mg/L (n=3)','#A63D40')]:ax.scatter(o.real[m],pred[m],label=label,color=color,s=50)
        ax.plot([2.8,6.3],[2.8,6.3],'--',color='#555555',label='1:1 agreement reference')
        ax.set(xlabel='Observed DO (mg/L)',ylabel='Out-of-fold predicted DO (mg/L)',title=name.replace('_',' '));ax.legend();save(fig,'OD_observed_predicted_'+name)
    err=pd.read_csv(OUT/'OD_fold_errors.csv');fig,ax=plt.subplots(figsize=(10,5));ax.bar(err.period if 'period' in err else err.iloc[:,0],err.RMSE,color='#245F88')
    ax.set(ylabel='RMSE (mg/L DO)',xlabel='Held-out campaign',title='Elastic Net error by held-out campaign');ax.tick_params(axis='x',rotation=55);save(fig,'OD_fold_errors')
    imp=pd.DataFrame(dr['permutation_importance'])
    for subset in ['all','without_consensus']:
        f=imp.loc[imp.subset.eq(subset)].sort_values('DeltaRMSE');fig,ax=plt.subplots(figsize=(9,6))
        ax.barh([SHORT[v] for v in f.variable],f.DeltaRMSE,xerr=f.SD_permutation,color='#245F88')
        ax.set(xlabel='Increase in out-of-fold RMSE (mg/L DO)',title='DO predictor permutation importance: '+subset.replace('_',' '));save(fig,'OD_importance_'+subset)
    flags=pd.read_csv(OUT/'anomaly_flags.csv');fig,ax=plt.subplots(figsize=(11,4.5));flag=flags.votes.ge(2)
    ax.scatter(np.arange(65)[~flag],flags.votes[~flag],color='#999999',s=25);ax.scatter(np.arange(65)[flag],flags.votes[flag],color='#A63D40',s=55)
    for period,f in flags.loc[flag].assign(period=flags.loc[flag,'ID'].str[:6]).groupby('period'):
        label=period[:4]+'-'+period[4:]+'\n'+', '.join(f.ID.str.split('_').str[1])
        ax.annotate(label,(float(f.index.to_numpy().mean()),float(f.votes.mean())),xytext=(0,12),
                    textcoords='offset points',ha='right' if period=='201703' else 'center',fontsize=10)
    ax.axhline(2,ls='--',color='#555555');ax.set(ylim=(-.2,4.1),yticks=[0,1,2,3],xlabel='Observations ordered by campaign and site',ylabel='Detector votes',title='Consensus anomalies (at least two of three detectors)');save(fig,'anomaly_consensus')
    cor=d[FEATURES].corr();fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(cor,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set(xticks=range(11),yticks=range(11),xticklabels=[SHORT[v] for v in FEATURES],yticklabels=[SHORT[v] for v in FEATURES],title='Pearson correlations (descriptive)');ax.tick_params(axis='x',rotation=65)
    for i in range(11):
        for j in range(11):ax.text(j,i,f'{cor.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(cor.iloc[i,j])>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Pearson r');save(fig,'correlations')
    print('Figures generated:',len(list(FIG.glob('*.png'))),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--variable',choices=VARS)
    main(parser.parse_args().variable)
