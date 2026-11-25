"""Gaussian identity GEE, with independently computed robust and MD covariance.

No source data are modified. Original observed-order estimates are cross-checked
against manuscript Table 3 before interpreting calendar-month sensitivities.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import brent
from scipy.stats import norm,t
from analysis_revision import load_data,VARS,LOG_VARS,OUT

def bh(p):
    p=np.asarray(p);idx=np.argsort(p);m=len(p)
    q=np.minimum.accumulate((p[idx]*m/np.arange(1,m+1))[::-1])[::-1]
    out=np.empty(m);out[idx]=np.minimum(q,1);return out

def update_rho(resids,times,previous):
    later=[];earlier=[];distance=[]
    for e,tt in zip(resids,times):
        i,j=np.tril_indices(len(e),-1)
        later.extend(e[i]);earlier.extend(e[j]);distance.extend(np.abs(tt[i]-tt[j]))
    later=np.array(later);earlier=np.array(earlier);distance=np.array(distance)
    variance=(1-previous**(2*distance))/(1-previous**2)
    weights=1/variance;weights/=weights.sum()
    def objective(a):return np.dot(weights,(later-a**distance*earlier)**2)
    midpoint=.5
    while objective(midpoint)>objective(0):
        midpoint/=2
        if midpoint<1e-8:return 0.
    upper=.75
    while objective(upper)<objective(midpoint):
        upper=(1+upper)/2
        if upper>1-1e-6:raise ValueError("AR rho optimum too close to one")
    return float(brent(objective,brack=(0.,midpoint,upper)))

def fit_gee(d,var,calendar=True,independence=False,compute_covariance=True):
    tt=d.Mes_transcurrido.to_numpy() if calendar else d.Orden_temporal.to_numpy(dtype=float)
    z=d.Zona_B.to_numpy()
    X=np.column_stack([np.ones(len(d)),z,tt,z*tt])
    y=d[var].to_numpy(dtype=float)
    if var in LOG_VARS:y=np.log1p(y)
    # Gaussian identity equations are invariant to affine response scaling.
    # Condition the numerical score so SDT (~20,000 mg/L) and trace nutrients
    # are tested with comparable convergence tolerances; restore all units below.
    y_center=float(y.mean());y_scale=float(y.std())
    y=(y-y_center)/y_scale
    blocks=[np.flatnonzero(d.Punto.to_numpy()==site) for site in sorted(d.Punto.unique())]
    xs=[X[ix] for ix in blocks];ys=[y[ix] for ix in blocks];ts=[tt[ix] for ix in blocks]
    rho=0.;beta=np.linalg.lstsq(X,y,rcond=None)[0];converged=False
    for iteration in range(500):
        ris=[]
        for a in ts:
            # Original Statsmodels implementation uses within-cluster row position
            # for the working matrix; calendar variant explicitly uses month gaps.
            a_cov=a if calendar else np.arange(len(a))
            R=rho**np.abs(a_cov[:,None]-a_cov[None,:])
            ris.append(np.linalg.inv(R))
        B=sum(x.T@r@x for x,r in zip(xs,ris))
        score=sum(x.T@r@(yy-x@beta) for x,r,yy in zip(xs,ris,ys))
        beta=beta+np.linalg.solve(B,score)
        if independence or (np.linalg.norm(score)<1e-6 and iteration>0):
            converged=True;break
        rho=update_rho([yy-x@beta for x,yy in zip(xs,ys)],ts,rho)
    residual=y-X@beta
    if not compute_covariance:
        beta_original=beta*y_scale;beta_original[0]+=y_center
        return [],dict(beta=beta_original.tolist(),rho=rho,converged=converged)
    binv=np.linalg.inv(B)
    meat=np.zeros((4,4));mdmeat=meat.copy();minlev=[]
    for x,yy,r in zip(xs,ys,ris):
        e=yy-x@beta
        u=x.T@r@e;meat+=np.outer(u,u)
        H=x@binv@x.T@r
        adj=np.linalg.solve(np.eye(len(e))-H,e)
        ua=x.T@r@adj;mdmeat+=np.outer(ua,ua)
        minlev.append(float(np.min(np.linalg.svd(np.eye(len(e))-H,compute_uv=False))))
    cov=binv@meat@binv;md=binv@mdmeat@binv
    contrasts={"A":np.array([0,0,1,0]),"B":np.array([0,0,1,1]),"B_minus_A":np.array([0,0,0,1])}
    rows=[]
    for label,c in contrasts.items():
        b=float(c@beta)*y_scale;se=float(np.sqrt(c@cov@c))*y_scale;semd=float(np.sqrt(c@md@c))*y_scale
        dof=len(blocks)-1;critical=t.ppf(.975,dof)
        rows.append(dict(variable=var,contrast=label,beta=b,se_robust=se,se_MD=semd,
            p_z_robust=float(2*norm.sf(abs(b/se))),p_z_MD=float(2*norm.sf(abs(b/semd))),
            p_t_MD=float(2*t.sf(abs(b/semd),dof)),df_t=dof,
            CI95_MD_t_low=float(b-critical*semd),CI95_MD_t_high=float(b+critical*semd),
            rho=rho,n=len(d),converged=converged,iterations=iteration+1,
            working_correlation="calendar_month" if calendar else "observed_order",
            transformed=(var in LOG_VARS)))
    fitted=X@beta
    beta_original=beta*y_scale;beta_original[0]+=y_center
    return rows,dict(beta=beta_original.tolist(),residuals=(residual*y_scale).tolist(),fitted=(fitted*y_scale+y_center).tolist(),
        R2_descriptive=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
        RMSE_descriptive=float(np.mean(residual**2)**.5*y_scale),min_I_minus_H_singular=min(minlev))

def main():
    d=load_data()
    # Original 2016/2017 identifiers, as confirmed by authors.
    anomaly_ids={"201611_P1","201610_P1","201611_P2","201610_P2","201703_P5","201604_P1"}
    sets={"all":d,"without_consensus":d.loc[~d.ID_muestra.isin(anomaly_ids)].reset_index(drop=True),
          "without_Oct_Nov_2016":d.loc[~d.Periodo.isin(["2016-10","2016-11"])].reset_index(drop=True)}
    rows=[];diagnostics={};errors=[]
    for name,ds in sets.items():
        for calendar in [False,True]:
            local=[]
            for var in VARS:
                try:
                    rr,diag=fit_gee(ds,var,calendar)
                    for r in rr:r["subset"]=name
                    local.extend(rr)
                    diagnostics[f"{name}_{calendar}_{var}"]=diag
                except Exception as e:errors.append(dict(subset=name,calendar=calendar,variable=var,error=str(e)))
            indices=[i for i,r in enumerate(local) if r["contrast"]=="B_minus_A"]
            for key in ["p_z_robust","p_z_MD","p_t_MD"]:
                adjusted=bh([local[i][key] for i in indices])
                for i,q in zip(indices,adjusted):local[i]["q_"+key]=float(q)
            rows.extend(local)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"gee_all_results.csv",index=False)
    (OUT/"gee_diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print("ERRORS",errors,flush=True)
    print("ORIGINAL ORDER REPRODUCTION",result.query("subset=='all' and working_correlation=='observed_order' and contrast!='B_minus_A'")[["variable","contrast","beta","rho","converged"]].to_string(index=False),flush=True)
    print("DO SENSITIVITY",result.query("variable=='OD_mg_L'")[["subset","working_correlation","contrast","beta","se_robust","se_MD","CI95_MD_t_low","CI95_MD_t_high","p_t_MD","q_p_t_MD","converged"]].to_string(index=False),flush=True)
    print("CALENDAR INTERACTIONS",result.query("subset=='all' and working_correlation=='calendar_month' and contrast=='B_minus_A'")[["variable","beta","q_p_z_robust","q_p_z_MD","q_p_t_MD"]].to_string(index=False),flush=True)
    # Algebraic check at working independence against direct least squares.
    rr,diag=fit_gee(d,"OD_mg_L",calendar=True,independence=True)
    tt=d.Mes_transcurrido.to_numpy();z=d.Zona_B.to_numpy();X=np.column_stack([np.ones(len(d)),z,tt,z*tt])
    assert np.allclose(diag["beta"],np.linalg.lstsq(X,d.OD_mg_L,rcond=None)[0],atol=1e-10)
    published={"Temperatura_C":(-.205,-.247),"pH":(-.020,-.032),"CE_mS_cm":(2.017,1.986),
               "SDT_mg_L":(1320.547,1328.458),"Salinidad_UPS":(1.389,1.378),"OD_mg_L":(-.049,-.006)}
    check=result.query("subset=='all' and working_correlation=='observed_order'")
    deltas=[]
    for var,bs in published.items():
        for contrast,b in zip(["A","B"],bs):
            actual=check.query("variable==@var and contrast==@contrast").iloc[0].beta
            deltas.append(dict(variable=var,contrast=contrast,expected_rounded=b,computed=float(actual),difference=float(actual-b)))
    print("MANUSCRIPT CHECK",deltas,flush=True)
    (OUT/"gee_validation.json").write_text(json.dumps(dict(errors=errors,manuscript_check=deltas),indent=2))

if __name__=="__main__":main()
