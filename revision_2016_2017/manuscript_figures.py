"""Recreate the manuscript's original panel layouts with verified 2016–2017 results.

Usage: python update_figures.py --results /path/to/results --output /path/to/figures
The supplied results directory contains the exact numeric inputs used here.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, log_loss)

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
R, F = args.results, args.output
F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN = '#1f77b4', '#ff7f0e', '#2ca02c'
FEATURES = ['Temperatura_C', 'pH', 'CE_mS_cm', 'OD_mg_L', 'DBO5_mg_L',
            'DQO_mg_L', 'SST_mg_L', 'SS_mL_L', 'NH3_N_mg_L', 'NO2_mg_L', 'NO3_mg_L']
SHORT = dict(zip(FEATURES, ['Temp.', 'pH', 'EC', 'DO', r'BOD$_5$', 'COD', 'TSS',
                         'SS', r'NH$_3$–N', r'NO$_2^-$', r'NO$_3^-$']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
    'axes.titlesize': 15, 'axes.labelsize': 14, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'legend.fontsize': 11, 'figure.titlesize': 18,
    'figure.titleweight': 'bold', 'axes.axisbelow': True, 'savefig.facecolor': 'white',
    'svg.fonttype': 'none'})

def load(name):
    return json.loads((R/name).read_text())

def grid(ax, axis='both'):
    ax.grid(True, axis=axis, color='#b0b0b0', alpha=.18, lw=.8)

def save(fig, number):
    number = {6: 7, 7: 8, 8: 9}.get(number, number)
    name = f'Figure_{number:02d}'
    fig.savefig(F/(name+'.png'), dpi=400, bbox_inches='tight', pad_inches=.12)
    fig.savefig(F/(name+'.svg'), bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    print('Created', name, flush=True)

d = pd.read_csv(R/'analysis_data_2016_2017.csv')
scores = pd.read_csv(R/'PCA_scores.csv')
loads = pd.read_csv(R/'PCA_loadings.csv', index_col=0)
a = load('analysis_results.json')
dr = load('diagnostic_results.json')
gd = load('gee_diagnostics.json')
assert scores.ID.tolist() == d.ID_muestra.tolist()
assert d.Periodo.nunique() == 13 and len(d) == 65
assert '2016-03' not in set(d.Periodo)

# Figure 2: retain the original 2 x 2 means-and-fitted-trends arrangement.
fig, axs = plt.subplots(2, 2, figsize=(13.5, 10))
fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.90, wspace=.22, hspace=.34)
fig.suptitle('Observed zone means and GEE–AR(1) temporal trends', y=.98)
variables = [('OD_mg_L', 'Dissolved oxygen', r'mg L$^{-1}$'),
             ('NO3_mg_L', 'Nitrate', r'mg L$^{-1}$'),
             ('DBO5_mg_L', r'BOD$_5$', r'mg L$^{-1}$'),
             ('CE_mS_cm', 'Electrical conductivity', r'mS cm$^{-1}$')]
tx = np.linspace(0, 13, 200)
for ax, (v, title, unit) in zip(axs.flat, variables):
    b = np.array(gd[f'all_True_{v}']['beta'])
    for z, color, marker in [(0, BLUE, 'o'), (1, ORANGE, 's')]:
        mean = d.loc[d.Zona_B.eq(z)].groupby('Mes_transcurrido')[v].mean()
        # The larger open circle preserves both markers where means coincide.
        ax.scatter(mean.index, mean.values, s=68 if z == 0 else 34, marker=marker,
                   facecolors='none' if z == 0 else color, edgecolors=color, lw=1.5,
                   label='Zone A (P1–P3)' if z == 0 else 'Zone B (P4–P5)', zorder=3)
        fit = b[0] + b[1]*z + (b[2]+b[3]*z)*tx
        if v in ['NO3_mg_L', 'DBO5_mg_L']:
            fit = np.expm1(fit)
        ax.plot(tx, fit, color=color, lw=2.2)
    ax.set(title=title, ylabel=unit, xlabel='Elapsed calendar month',
           xticks=[0, 2, 4, 6, 8, 10, 12, 13])
    ax.set_xlim(-.5, 13.5)
    grid(ax)
axs[0, 0].legend(loc='lower left', frameon=False)
axs[0, 0].annotate('Apr 2016: A = B', (2, 5.89), xytext=(3.7, 5.81), fontsize=10,
                   arrowprops={'arrowstyle': '-', 'color': '#666666'})
save(fig, 2)

# Figure 3: restore biplot with arrows and variance/cumulative-variance panel.
fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
fig.subplots_adjust(left=.06, right=.94, bottom=.13, top=.83, wspace=.29)
fig.suptitle('Principal component analysis of standardized physicochemical profiles', y=.98, fontsize=17)
ax = axs[0]
for zone, color, marker in [('A', BLUE, 'o'), ('B', ORANGE, 's')]:
    f = scores.loc[scores.zone.eq(zone)]
    ax.scatter(f.PC1, f.PC2, color=color, marker=marker, s=34, alpha=.75,
               label=f'Zone {zone} (P1–P3)' if zone == 'A' else 'Zone B (P4–P5)')
ax.axhline(0, color='#888888', lw=.8); ax.axvline(0, color='#888888', lw=.8)
label_positions = {
    'CE_mS_cm': (-.3, 2.10), 'NO2_mg_L': (1.7, .90), 'NH3_N_mg_L': (2.2, .47),
    'NO3_mg_L': (2.9, .12), 'DQO_mg_L': (2.8, -.24), 'DBO5_mg_L': (2.6, -.65),
    'SST_mg_L': (2.0, -1.00), 'SS_mL_L': (.95, -1.80), 'Temperatura_C': (-.05, -1.40),
    'pH': (-.7, -1.72), 'OD_mg_L': (-1.45, -.70)}
for v in FEATURES:
    xy = loads.loc[v, ['PC1', 'PC2']].to_numpy()*3
    ax.annotate('', xy=xy, xytext=(0, 0), arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1})
    ax.annotate(SHORT[v], xy=xy, xytext=label_positions[v], fontsize=10, ha='center', va='center',
                arrowprops={'arrowstyle': '-', 'color': '#777777', 'lw': .6},
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .75, 'pad': .4})
ax.set(title='A. Scores and variable loadings', xlabel='PC1 (54.8%)', ylabel='PC2 (20.2%)',
       xlim=(-4.8, 9.8), ylim=(-2.4, 3.6))
ax.legend(loc='lower right', framealpha=.9, fontsize=10); grid(ax)
ax = axs[1]; vr = np.array(a['pca']['variance_ratio'])*100; xx = np.arange(1, 12)
ax.bar(xx, vr, color=BLUE, alpha=.9)
ax.set(title='B. Explained and cumulative variance', xlabel='Principal component',
       ylabel='Explained variance (%)', xticks=xx, ylim=(0, 58))
ax.set_xticklabels([f'PC{i}' for i in xx], rotation=45, ha='right', fontsize=10)
grid(ax, 'y')
right = ax.twinx(); right.plot(xx, np.cumsum(vr), 'o-', color=ORANGE, lw=2)
right.axhline(80, color='#888888', ls='--', lw=1)
right.set(ylabel='Cumulative variance (%)', ylim=(0, 105))
save(fig, 3)

# Figure 4: restore all three original clustering panels and cluster colors.
Z = StandardScaler().fit_transform(d[FEATURES])
km = KMeans(n_clusters=3, n_init=50, random_state=42).fit(Z)
raw_sizes = np.bincount(km.labels_)
mapping = {int(np.flatnonzero(raw_sizes == n)[0]): k for k, n in [(1, 41), (2, 20), (3, 4)]}
clusters = np.array([mapping[v] for v in km.labels_])
assert np.bincount(clusters)[1:].tolist() == [41, 20, 4]
pd.DataFrame({'ID': d.ID_muestra, 'cluster': clusters}).to_csv(F/'cluster_membership.csv', index=False)
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.7), gridspec_kw={'width_ratios': [1, 1.23, .83]})
fig.subplots_adjust(left=.055, right=.99, bottom=.23, top=.80, wspace=.30)
fig.suptitle('Unsupervised clustering of physicochemical water-quality profiles', y=.97, fontsize=17)
for k, color in [(1, BLUE), (2, ORANGE), (3, GREEN)]:
    ix = clusters == k
    axs[0].scatter(scores.loc[ix, 'PC1'], scores.loc[ix, 'PC2'], color=color, s=32, alpha=.85, label=f'Cluster {k}')
    axs[1].plot(np.arange(11), Z[ix].mean(axis=0), 'o-', color=color, lw=1.7, ms=4, label=f'Cluster {k}')
axs[0].set(title='A. Cluster assignments in PCA space', xlabel='PC1', ylabel='PC2')
axs[0].legend(loc='lower right', fontsize=10)
axs[1].set(title='B. Standardized physicochemical profiles', ylabel='Standardized cluster mean', xticks=range(11))
axs[1].set_xticklabels([SHORT[v] for v in FEATURES], rotation=55, ha='right', fontsize=11)
axs[1].legend(loc='upper left', fontsize=10)
axs[2].plot([r['k'] for r in a['clusters']], [r['silhouette'] for r in a['clusters']], 'o-', color=BLUE)
axs[2].set(title='C. Cluster-number assessment', xlabel='Number of clusters (k)', ylabel='Mean silhouette width',
           xticks=[2, 3, 4], ylim=(.30, .37))
for ax in axs: grid(ax)
save(fig, 4)

# Figure 5: metrics, confusion matrix and ROC curves, in their original order.
names = ['Logistic', 'SVM', 'RF']; display = ['L2 logistic\nregression', 'Linear SVM', 'Random forest']
metrics = []; clfs = {}
for name in names:
    x = pd.read_csv(R/f'clf_Periodo_{name}.csv'); clfs[name] = x
    metrics.append([balanced_accuracy_score(x.real, x.prob >= .5),
                    f1_score(x.real, x.prob >= .5, average='macro'), roc_auc_score(x.real, x.prob)])
metrics = np.array(metrics)
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.5), gridspec_kw={'width_ratios': [1.08, .85, 1.08]})
fig.subplots_adjust(left=.045, right=.99, bottom=.16, top=.79, wspace=.29)
fig.suptitle('Exploratory classification of predefined site groups', y=.97)
for j, (label, color) in enumerate(zip(['Balanced accuracy', 'Macro-F1', 'ROC-AUC'], [BLUE, ORANGE, GREEN])):
    axs[0].bar(np.arange(3)+(j-1)*.25, metrics[:, j], width=.25, color=color, label=label)
axs[0].axhline(.5, color='#888888', ls='--', lw=1)
axs[0].set(title='A. Grouped cross-validation performance', ylabel='Score', xticks=range(3), xticklabels=display, ylim=(0, 1.3), yticks=[0, .2, .4, .6, .8, 1])
axs[0].legend(loc='upper left', fontsize=10); grid(axs[0], 'y')
cf = confusion_matrix(clfs['Logistic'].real, clfs['Logistic'].prob >= .5)
assert cf.tolist() == [[24, 2], [6, 33]]
axs[1].imshow(cf, cmap='Blues', vmin=0, vmax=33, aspect='auto')
for i in range(2):
    for j in range(2): axs[1].text(j, i, str(cf[i, j]), ha='center', va='center', fontsize=22, color='white' if cf[i, j] > 16 else 'black')
axs[1].set(title='B. Out-of-fold confusion matrix', xlabel='Predicted', ylabel='Observed',
           xticks=[0, 1], yticks=[0, 1], xticklabels=['Zone B', 'Zone A'], yticklabels=['Zone B', 'Zone A'])
for name, color, label in zip(names, [BLUE, ORANGE, GREEN], ['Logistic', 'Linear SVM', 'Random forest']):
    x = clfs[name]; fp, tp, _ = roc_curve(x.real, x.prob)
    axs[2].plot(fp, tp, color=color, lw=1.8, label=f'{label} (AUC {roc_auc_score(x.real, x.prob):.3f})')
axs[2].plot([0, 1], [0, 1], '--', color='#888888', lw=1)
axs[2].set(title='C. Out-of-fold ROC curves', xlabel='False-positive rate', ylabel='True-positive rate')
axs[2].legend(loc='lower right', fontsize=10); grid(axs[2])
save(fig, 5)

# Figure 6: original horizontal-RMSE comparison and observed/predicted scatter.
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(left=.13, right=.99, bottom=.15, top=.80, wspace=.32)
fig.suptitle('Dissolved-oxygen prediction under leave-one-campaign-out validation', y=.97, fontsize=17)
oof = pd.read_csv(R/'oof_Periodo_ElasticNet.csv')
baseline = float(np.mean((oof.real-oof.baseline)**2)**.5)
nest = pd.read_csv(R/'nested_oof.csv'); nested_rmse = float(np.mean((oof.real-nest.predicted)**2)**.5)
reg = {r['model']: r['RMSE'] for r in a['regression'] if r['group'] == 'Periodo'}
bars = [('Training-mean\nbaseline', baseline, BLUE), ('SVR', reg['SVR'], BLUE),
        ('Nested model\nselection', nested_rmse, ORANGE), ('Ridge', reg['Ridge'], BLUE),
        ('Random forest', reg['RF'], BLUE), ('Elastic Net', reg['ElasticNet'], BLUE)]
axs[0].barh(range(len(bars)), [r[1] for r in bars], color=[r[2] for r in bars])
axs[0].invert_yaxis(); axs[0].set(yticks=range(len(bars)), yticklabels=[r[0] for r in bars], xlim=(0, .70),
    xlabel=r'Out-of-fold RMSE (mg L$^{-1}$)', title='A. Comparison with baseline')
for i, (_, value, _) in enumerate(bars): axs[0].text(value+.006, i, f'{value:.3f}', va='center', fontsize=11)
grid(axs[0], 'x')
axs[1].scatter(oof.real, oof.pred, color=BLUE, s=39, alpha=.75)
axs[1].plot([2.9, 6.2], [2.9, 6.2], '--', color='#666666', lw=1.5, label='1:1 agreement reference')
axs[1].set(title='B. Elastic Net out-of-fold predictions', xlabel=r'Observed DO (mg L$^{-1}$)', ylabel=r'Predicted DO (mg L$^{-1}$)')
axs[1].legend(loc='lower right', fontsize=10)
axs[1].text(.04, .94, r'$R^2$ = 0.385; RMSE = 0.448', transform=axs[1].transAxes, va='top', fontsize=11)
grid(axs[1]); save(fig, 6)

# Restore classification permutation importance using the original 30-repeat protocol.
X = d[FEATURES]; y = (d.Punto <= 3).astype(int).to_numpy()
prob = np.empty(len(d)); folds = []
for train, test in LeaveOneGroupOut().split(X, y, d.Periodo):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=5000, random_state=42)).fit(X.iloc[train], y[train])
    prob[test] = m.predict_proba(X.iloc[test])[:, 1]; folds.append((m, test))
assert np.allclose(prob, clfs['Logistic'].prob, atol=1e-12)
base_loss = log_loss(y, prob); rng = np.random.default_rng(42); importance = []
for variable in FEATURES:
    deltas = []
    for repeat in range(30):
        shuffled_prob = prob.copy()
        for m, test in folds:
            xt = X.iloc[test].copy(); xt[variable] = rng.permutation(xt[variable].to_numpy())
            shuffled_prob[test] = m.predict_proba(xt)[:, 1]
        deltas.append(log_loss(y, shuffled_prob)-base_loss)
    importance.append({'variable': variable, 'DeltaLogLoss': float(np.mean(deltas)), 'SD_permutation': float(np.std(deltas, ddof=1))})
ci = pd.DataFrame(importance); ci.to_csv(F/'classification_permutation_importance.csv', index=False)
di = pd.DataFrame(dr['permutation_importance']); di = di.loc[di.subset.eq('all')]
fig, axs = plt.subplots(1, 2, figsize=(14, 6.4))
fig.subplots_adjust(left=.11, right=.98, bottom=.14, top=.82, wspace=.39)
fig.suptitle('Grouped out-of-sample permutation importance', y=.97)
for ax, frame, metric, title, xlabel in [
    (axs[0], ci, 'DeltaLogLoss', 'A. Exploratory site-group classification', 'Increase in log loss after permutation'),
    (axs[1], di, 'DeltaRMSE', 'B. Dissolved-oxygen estimation', r'Increase in RMSE after permutation (mg L$^{-1}$)')]:
    frame = frame.sort_values(metric, ascending=False)
    ax.barh(range(len(frame)), frame[metric], xerr=frame.SD_permutation, color=BLUE, capsize=2)
    ax.set(yticks=range(len(frame)), yticklabels=[SHORT[v] for v in frame.variable], title=title, xlabel=xlabel)
    ax.invert_yaxis(); ax.axvline(0, color='#777777', lw=.8); grid(ax, 'x')
save(fig, 7)

# Figure 8: original Isolation Forest scores and consensus flags in PCA space.
flags = pd.read_csv(R/'anomaly_flags.csv'); assert flags.ID.tolist() == d.ID_muestra.tolist()
rz = RobustScaler().fit_transform(d[FEATURES])
forest = IsolationForest(n_estimators=500, contamination=.1, random_state=42).fit(rz)
if_score = -forest.score_samples(rz)
assert np.array_equal(forest.predict(rz) == -1, flags['IF'].to_numpy())
is_flag = flags.votes.to_numpy() >= 2
pd.DataFrame({'ID': d.ID_muestra, 'IsolationForest_score': if_score, 'consensus': is_flag}).to_csv(F/'anomaly_scores.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(14, 6.4))
fig.subplots_adjust(left=.07, right=.98, bottom=.28, top=.81, wspace=.20)
fig.suptitle('Consensus multivariate anomaly detection', y=.97)
for ax, xx, yy in [(axs[0], np.arange(65), if_score), (axs[1], scores.PC1.to_numpy(), scores.PC2.to_numpy())]:
    ax.scatter(xx[~is_flag], yy[~is_flag], color=BLUE, s=30, label='Not consensus anomaly')
    ax.scatter(xx[is_flag], yy[is_flag], color=ORANGE, marker='D', s=70, label='Consensus anomaly')
    grid(ax)
axs[0].set(title='A. Anomaly score by sample', ylabel='Isolation Forest score', xticks=np.arange(0, 65, 5))
axs[0].set_xticklabels(d.ID_muestra.iloc[::5], rotation=65, ha='right', fontsize=10)
axs[0].legend(loc='upper left', fontsize=10, framealpha=.9)
axs[1].set(title='B. Consensus anomalies in PCA space', xlabel='PC1', ylabel='PC2', xlim=(-4.8, 10.8), ylim=(-2.4, 3.6))
offsets = {'201604_P1': (-68, -5), '201610_P1': (5, -14), '201610_P2': (-60, 10),
           '201611_P1': (-65, 12), '201611_P2': (5, 9), '201703_P5': (5, 9)}
for ix in np.flatnonzero(is_flag):
    ident = d.ID_muestra.iloc[ix]
    axs[1].annotate(ident, (scores.PC1.iloc[ix], scores.PC2.iloc[ix]), xytext=offsets[ident],
                    textcoords='offset points', fontsize=9)
axs[1].legend(loc='lower right', fontsize=10)
save(fig, 8)

summary = {'main_figures_restored': list(range(2, 9)), 'map_unchanged': True,
    'source_period': ['2016-02', '2017-03'], 'no_sampling': '2016-03',
    'classification_importance_top3': ci.sort_values('DeltaLogLoss', ascending=False).head(3).to_dict('records'),
    'classification_OOF_reproduced': True, 'anomaly_IF_flags_reproduced': True,
    'cluster_sizes': [41, 20, 4], 'confusion_matrix': cf.tolist()}
(F/'figure_numeric_checks.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)

# Integrated main-text figures: calibration (6) and detector votes (10).
import shutil
for source, target in [('classifier_calibration', 'Figure_06'), ('anomaly_consensus', 'Figure_10')]:
    for suffix in ('.png', '.svg'):
        shutil.copy2(R.parent/'figures'/(source+suffix), F/(target+suffix))
