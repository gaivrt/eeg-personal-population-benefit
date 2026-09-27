"""One-command stage 0a rerun; no SSH, GPU, or later-stage entry point."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")


def call(*args):
    subprocess.run([sys.executable, *args], check=True)


parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true", help="Fetch only the fixed single-subject audit manifest")
parser.add_argument("--config", default="configs/stage0a.yaml")
args = parser.parse_args()
if args.download:
    call("scripts/fetch_assets.py")
call("-m", "pytest", "-q", "--junitxml=reports/stage0a/tests.xml")
call("scripts/backbone_smoke.py")
call("scripts/cache_ablation.py")
call("-m", "subject_context.run", "--config", args.config)
call("scripts/audit_samples.py")
call("scripts/resource_estimate.py")
print("Stage 0a ended. No stage 0b/A/B/C/D work is scheduled.")
