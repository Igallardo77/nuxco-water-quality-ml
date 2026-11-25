"""Run all analyses in the order required by their dependencies."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ['analysis_revision.py','gee_revision.py','diagnostics_revision.py','extended_checks.py','make_figures.py','build_report.py']:
    print('Running',name,flush=True)
    subprocess.run([sys.executable,'-u',str(root/name)],cwd=root,check=True)
print('Complete: Resultados_revision_Nuxco.md, results/ and figures/')

subprocess.run([sys.executable, str(root/"manuscript_figures.py"), "--results", str(root/"results"), "--output", str(root.parent/"figures/manuscript")], cwd=root, check=True)
