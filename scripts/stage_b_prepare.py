"""Freeze temporal halves using existing metadata only, without EEG processing."""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.stage_a_common import ROOT, save_json, sha256
from subject_context.stage_b import temporal_halves

p=argparse.ArgumentParser();p.add_argument("--processed",type=Path,required=True)
args=p.parse_args()
people=json.loads((ROOT/"configs/splits/stage_a_v1.json").read_text())["subjects"]
records=[];counts=[]
for dataset,subject in people:
    path=args.processed/dataset/f"sub-{subject:03d}"/"trials.csv"
    trials=pd.read_csv(path)
    fit,query,row=temporal_halves(trials)
    records.append({"dataset":dataset,"subject":subject,"trial_table_sha256":sha256(path),
                    "fit_indices":fit.tolist(),"query_indices":query.tolist(),
                    "order_basis":"published_sequence_and_original_event" if dataset=="Cho2017" else "protocol_run_then_original_onset"})
    counts.append({"dataset":dataset,"subject":subject,**row,"fit_n":len(fit),"query_n":len(query)})
out=ROOT/"configs/splits/stage_b_halves_v1.json"
assert not out.exists(), "Temporal halves already frozen; do not overwrite"
split_bytes=subprocess.check_output(["git","show","HEAD:configs/splits/stage_a_v1.json"],cwd=ROOT)
save_json(out,{"version":1,"split_sha256":hashlib.sha256(split_bytes).hexdigest(),"subjects":records})
report=ROOT/"reports/stage_b";report.mkdir(exist_ok=True,parents=True)
pd.DataFrame(counts).to_csv(report/"temporal_balance.csv",index=False)
print(pd.DataFrame(counts).groupby("dataset")[["fit_n","query_n"]].agg(["min","max"]))
print("Imbalanced halves:\n",pd.DataFrame(counts).query("not fit_balanced or not query_balanced").to_string(index=False))
