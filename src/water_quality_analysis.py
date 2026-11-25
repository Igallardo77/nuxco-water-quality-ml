"""Entry point for the revised 2016-2017 analysis."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "revision_2016_2017/run_revision.py")], cwd=root, check=True)
