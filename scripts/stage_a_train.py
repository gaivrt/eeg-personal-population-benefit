"""Fixed subject folds, validation-only selection, one RTX 3090 per Slurm task."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from pyriemann.tangentspace import TangentSpace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.data import assert_split
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256

parser=argparse.ArgumentParser()
parser.add_argument("--job",type=int,required=True)
parser.add_argument("--condition",choices=["B0","B1a","B1b","B1c","classic"],required=True)
args=parser.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(4)
cfg=config();root=scratch_root()
fold_index=args.job if args.condition=="classic" else args.job//5
seeds=cfg["seeds"] if args.condition=="classic" else [cfg["seeds"][args.job%5]]
fold=json.loads((ROOT/"configs/splits/stage_a_v1.json").read_text())["folds"][fold_index]
assert_split(*[[tuple(p) for p in fold[k]] for k in ("train","validation","test")])
receipt=provenance();receipt.update(gpu=torch.cuda.get_device_name(),condition=args.condition,fold=fold_index,
    split_sha256=sha256(ROOT/"configs/splits/stage_a_v1.json"),training_budget=cfg["training"],runs=[])
started=time.monotonic();torch.cuda.reset_peak_memory_stats()

def read_split(name):
    arrays=[];labels=[];ids=[];trial_ids=[]
    for index,(dataset,subject) in enumerate(fold[name]):
        directory=root/"features_v2"/dataset/f"sub-{subject:03d}"
        assert (directory/"done.json").exists()
        if args.condition=="B0":filename="b0"
        elif args.condition=="B1c":filename="ea_eo"
        elif args.condition=="classic":filename="classic_all" if name=="train" else "classic_eo"
        else:filename="ea_all" if name=="train" else ("ea_eo" if args.condition=="B1a" else "ea_task")
        x=np.load(directory/f"{filename}.npy")
        y=np.load(directory/"labels.npy")
        assert len(x)==len(y)
        arrays.append(x);labels.append(y);ids.extend([index]*len(y))
        trial_ids.extend(pd.read_csv(root/"processed"/dataset/f"sub-{subject:03d}"/"trials.csv")["trial_id"])
    return np.concatenate(arrays),np.concatenate(labels),np.array(ids),trial_ids

data={k:read_split(k) for k in ("train","validation","test")}
if args.condition=="classic":
    tangent=TangentSpace(metric="riemann")
    tangent.fit(data["train"][0])
    data={k:(tangent.transform(v[0]),*v[1:]) for k,v in data.items()}
    geometry_seconds=time.monotonic()-started
else:geometry_seconds=0
mean=data["train"][0].mean(0,dtype=np.float64)
std=data["train"][0].std(0,dtype=np.float64)
std=np.maximum(std,1e-6)
dtype=torch.float64 if args.condition=="classic" else torch.float32
features={k:torch.as_tensor((v[0]-mean)/std,device="cuda",dtype=dtype) for k,v in data.items()}
labels={k:torch.as_tensor(v[1],device="cuda",dtype=torch.long) for k,v in data.items()}
subject_ids={k:torch.as_tensor(v[2],device="cuda",dtype=torch.long) for k,v in data.items()}

def person_ba(pred,name):
    bins=subject_ids[name]*2+labels[name]
    count=torch.bincount(bins,minlength=2*len(fold[name]))
    correct=torch.bincount(bins,weights=(pred==labels[name]).double(),minlength=len(count))
    assert (count>0).all()
    return (correct/count).reshape(-1,2).mean(1)

@torch.no_grad()
def score(model,name,classic=False):
    z=model(features[name])
    pred=(z[:,0]>0).long() if classic else z.argmax(1)
    return float(person_ba(pred,name).mean()),pred

def save_run(model,head,seed,selection,trace,seconds,classic=False):
    tag="classic" if classic else args.condition+"_"+head
    out=root/"baselines_v2"/tag/f"fold-{fold_index}_seed-{seed}"
    out.mkdir(parents=True,exist_ok=True)
    _,pred=score(model,"test",classic)
    ba=person_ba(pred,"test").cpu().numpy()
    rows=[{"condition":tag,"fold":fold_index,"seed":seed,"dataset":d,"subject":s,
           "balanced_accuracy":float(value),"test_trials":int((data["test"][2]==i).sum())}
          for i,((d,s),value) in enumerate(zip(fold["test"],ba))]
    pd.DataFrame(rows).to_csv(out/"subjects.csv",index=False)
    pd.DataFrame({"trial_id":data["test"][3],"label":data["test"][1],"prediction":pred.cpu().numpy()}).to_csv(out/"predictions.csv",index=False)
    # Compact head checkpoint; the shared frozen backbone is never copied or trained.
    torch.save({"head":{k:v.cpu() for k,v in model.state_dict().items()},"feature_mean":mean,
                "feature_std":std,"selection":selection},out/"head.pt")
    result={**receipt,"head":head,"seed":seed,"selection":selection,"validation_search":trace,
            "seconds":seconds,"geometry_seconds":geometry_seconds,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),
            "n_train_trials":len(labels["train"]),"n_validation_trials":len(labels["validation"]),
            "n_test_trials":len(labels["test"])}
    result.pop("runs",None)
    save_json(out/"run.json",result)
    receipt["runs"].append({"condition":tag,"seed":seed,"selection":selection,"seconds":seconds})
    print(json.dumps({"condition":tag,"fold":fold_index,"seed":seed,"selection":selection,"seconds":seconds}),flush=True)

for seed in seeds:
    if args.condition=="classic":
        begin=time.monotonic();best=-1;trace=[]
        for c in [.1,1.,10.]:
            torch.manual_seed(seed)
            model=nn.Linear(features["train"].shape[1],1).to(device="cuda",dtype=torch.float64)
            nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
            optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=200,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                loss=nn.functional.binary_cross_entropy_with_logits(model(features["train"])[:,0],labels["train"].double())
                loss=loss+.5*model.weight.square().sum()/(c*len(labels["train"]))
                loss.backward();return loss
            optimizer.step(closure)
            value,_=score(model,"validation",True)
            trace.append({"C":c,"validation_subject_ba":value})
            if value>best:
                best=value;chosen=copy.deepcopy(model.state_dict());selection={"C":c,"validation_subject_ba":value}
        model.load_state_dict(chosen)
        save_run(model,"logistic",seed,selection,trace,time.monotonic()-begin,True)
        continue
    for head in cfg["training"]["heads"]:
        begin=time.monotonic();best=-1;trace=[]
        for lr in cfg["training"]["learning_rates"]:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            width=features["train"].shape[1]
            model=(nn.Linear(width,2) if head=="linear" else nn.Sequential(nn.Linear(width,cfg["training"]["mlp_hidden"]),nn.GELU(),nn.Linear(cfg["training"]["mlp_hidden"],2))).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg["training"]["weight_decay"])
            for epoch in range(1,cfg["training"]["epochs"]+1):
                model.train()
                order=torch.randperm(len(labels["train"]),device="cuda")
                for first in range(0,len(order),cfg["training"]["batch_size"]):
                    take=order[first:first+cfg["training"]["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    loss=nn.functional.cross_entropy(model(features["train"][take]),labels["train"][take])
                    loss.backward();optimizer.step()
                model.eval();value,_=score(model,"validation")
                trace.append({"learning_rate":lr,"epoch":epoch,"validation_subject_ba":value})
                if value>best:
                    best=value;chosen=copy.deepcopy(model.state_dict());selection={"learning_rate":lr,"epoch":epoch,"validation_subject_ba":value}
        model.load_state_dict(chosen)
        save_run(model,head,seed,selection,trace,time.monotonic()-begin)
receipt.update(status="complete",seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
save_json(root/f"receipts/train-{args.condition}-{args.job}.json",receipt)
