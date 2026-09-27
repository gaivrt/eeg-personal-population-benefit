"""One fixed outer fold/seed, validation selection then personal FiLM oracle."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.data import assert_split
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256
from subject_context.stage_b import config_b, temporal_halves, Readout, DirectAdapter, ba, fit_path

p=argparse.ArgumentParser();p.add_argument("--job",type=int,required=True)
p.add_argument("--benchmark",action="store_true");args=p.parse_args()
require_compute();assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4);cfg=config_b();root=scratch_root()
fold_index=args.job//5;seed=cfg["seeds"][args.job%5]
fold=json.loads((ROOT/cfg["split_file"]).read_text())["folds"][fold_index]
assert_split(*[[tuple(x) for x in fold[k]] for k in ["train","validation","test"]])
halves_path=ROOT/"configs/splits/stage_b_halves_v1.json"
halves={(r["dataset"],r["subject"]):r for r in json.loads(halves_path.read_text())["subjects"]}
out=root/"stage_b"/("benchmark" if args.benchmark else "runs")/f"fold-{fold_index}_seed-{seed}"
out.mkdir(parents=True,exist_ok=True)
receipt=provenance();receipt.update(stage_b_config_sha256=sha256(ROOT/"configs/stage_b.yaml"),
    split_sha256=sha256(ROOT/cfg["split_file"]),halves_sha256=sha256(halves_path),gpu=torch.cuda.get_device_name(),fold=fold_index,seed=seed)
start=time.monotonic();torch.cuda.reset_peak_memory_stats()
base=root/"baselines_v2"
scores={head:json.loads((base/f"B0_{head}"/f"fold-{fold_index}_seed-{seed}"/"run.json").read_text())["selection"]["validation_subject_ba"] for head in ["linear","mlp"]}
kind=max(scores,key=scores.get);checkpoint=base/f"B0_{kind}"/f"fold-{fold_index}_seed-{seed}"/"head.pt"
readout=Readout(torch.load(checkpoint,map_location="cpu",weights_only=False),kind).cuda()
backbone=load_backbone(ROOT/config()["model"]["weights"],k=4).cuda().eval()
assert sha256(ROOT/config()["model"]["weights"])==config()["model"]["weights_sha256"]
backbone_before=state_hash(backbone);head_before=state_hash(readout)
common=json.loads((ROOT/"configs/stage_a_channels.json").read_text())["readout_common"]
receipt.update(base_head=kind,base_head_validation_scores=scores,base_checkpoint_sha256=sha256(checkpoint))
zero_checks=[]

def load_subject(dataset,subject):
    src=root/"processed"/dataset/f"sub-{subject:03d}";f=root/"features_v2"/dataset/f"sub-{subject:03d}"
    record=halves[(dataset,subject)]
    assert sha256(src/"trials.csv")==record["trial_table_sha256"]
    trials=pd.read_csv(src/"trials.csv");fit,query,counts=temporal_halves(trials)
    assert fit.tolist()==record["fit_indices"] and query.tolist()==record["query_indices"]
    y=np.load(f/"labels.npy");np.testing.assert_array_equal(y,trials.label)
    meta=json.loads((src/"metadata.json").read_text());picks=[meta["channels"].index(c) for c in common]
    p8=torch.as_tensor(np.load(f/"prefix_tasks.npy").astype(np.float32),device="cuda")
    b0=torch.as_tensor(np.load(f/"b0.npy"),device="cuda")
    with torch.no_grad():
        p10=[]
        for chunk in p8.split(32):
            for layer in backbone.model.encoder.layers[-4:-2]:chunk=layer(chunk)
            p10.append(chunk)
        p10=torch.cat(p10)
        baseline=torch.cat([readout(chunk) for chunk in b0.split(32)])
        for variant in cfg["variants"]:
            adapter=DirectAdapter(backbone,readout,picks,variant).cuda()
            x=b0 if variant=="head" else (p10 if variant=="film2" else p8)
            zero=torch.cat([adapter(chunk) for chunk in x.split(32)])
            torch.testing.assert_close(zero,baseline,rtol=1e-4,atol=1e-4)
            assert torch.equal(zero.argmax(1),baseline.argmax(1)), "zero FiLM changes B0 predictions"
            zero_checks.append({"dataset":dataset,"subject":subject,"variant":variant,"max_logit_error":float((zero-baseline).abs().max())})
        baseline_pred=baseline.argmax(1).cpu().numpy()[query]
    return {"p8":p8,"p10":p10,"b0":b0,"picks":picks,"fit":torch.tensor(fit,device="cuda"),
            "query":torch.tensor(query,device="cuda"),"labels":torch.tensor(y,device="cuda"),
            "query_labels":y[query],"b0_ba":ba(y[query],baseline_pred),"b0_pred":baseline_pred,
            "trial_ids":trials.trial_id.iloc[query].tolist(),"counts":counts}

validation=fold["validation"]
if args.benchmark:
    validation=[next(p for p in validation if p[0]==d) for d in config()["datasets"]]
rows=[];paths=0;updates=0
for dataset,subject in validation:
    s=load_subject(dataset,subject)
    for variant in cfg["variants"]:
        lrs=cfg["head_learning_rates" if variant=="head" else "learning_rates"]
        for lr in (lrs[:1] if args.benchmark else lrs):
            for wd in (cfg["weight_decays"][:1] if args.benchmark else cfg["weight_decays"]):
                steps=[20] if args.benchmark else cfg["steps"]
                result,_=fit_path(backbone,readout,s,variant,lr,wd,steps,seed+subject)
                paths+=1;updates+=max(steps)
                for step,r in result.items():rows.append({"dataset":dataset,"subject":subject,"variant":variant,
                    "learning_rate":lr,"weight_decay":wd,"steps":step,"b0_ba":s["b0_ba"],"ba":r["ba"],"gain":r["ba"]-s["b0_ba"]})
    print(json.dumps({"phase":"validation","dataset":dataset,"subject":subject,"seconds":time.monotonic()-start}),flush=True)
    del s
pd.DataFrame(rows).to_csv(out/"validation_search.csv",index=False)
if not args.benchmark:
    from subject_context.stage_b import select_candidate
    selected={v:select_candidate([r for r in rows if r["variant"]==v]) for v in cfg["variants"]}
    winner=max(cfg["variants"][:-1],key=lambda v:(selected[v]["validation_median_gain"],selected[v]["validation_mean_gain"]))
    save_json(out/"selection.json",{"base_head":kind,"parameters":selected,"selected_film":winner})
    subjects=[];predictions=[]
    for dataset,subject in fold["test"]:
        s=load_subject(dataset,subject);params={}
        for variant in cfg["variants"]:
            hp=selected[variant]
            result,raw=fit_path(backbone,readout,s,variant,hp["learning_rate"],hp["weight_decay"],[hp["steps"]],seed+subject)
            paths+=1;updates+=hp["steps"];r=result[hp["steps"]]
            subjects.append({"dataset":dataset,"subject":subject,"fold":fold_index,"seed":seed,"base_head":kind,"variant":variant,
                "selected_film":variant==winner,"b0_ba":s["b0_ba"],"ba":r["ba"],"gain":r["ba"]-s["b0_ba"],**s["counts"]})
            for tid,y,b,pred in zip(s["trial_ids"],s["query_labels"],s["b0_pred"],r["predictions"]):
                predictions.append({"dataset":dataset,"subject":subject,"fold":fold_index,"seed":seed,"variant":variant,
                    "trial_id":tid,"label":int(y),"b0_prediction":int(b),"prediction":int(pred)})
            if raw is not None:params[variant]=raw
        torch.save(params,out/f"film-{dataset}-{subject:03d}.pt")
        pd.DataFrame(subjects).to_csv(out/"subjects.csv",index=False)
        print(json.dumps({"phase":"test","dataset":dataset,"subject":subject,"seconds":time.monotonic()-start}),flush=True)
        del s
    pd.DataFrame(predictions).to_csv(out/"predictions.csv",index=False)
assert state_hash(backbone)==backbone_before and state_hash(readout)==head_before
assert all(p.grad is None for p in backbone.parameters())
save_json(out/"zero_checks.json",zero_checks)
receipt.update(status="complete",seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
               optimization_paths=paths,optimization_steps=updates,backbone_unchanged=True,base_head_unchanged=True)
save_json(out/"run.json",receipt);print(json.dumps(receipt),flush=True)
