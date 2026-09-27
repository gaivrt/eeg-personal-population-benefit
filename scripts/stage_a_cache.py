"""Frozen CBraMod prefix-8 cache and reusable baseline features, one GPU per worker."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.model import load_backbone
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256
from subject_context.stage_a_features import ea_references, classic_covariances

parser=argparse.ArgumentParser()
parser.add_argument("--worker",type=int,default=0)
parser.add_argument("--workers",type=int,default=8)
parser.add_argument("--limit",type=int)
args=parser.parse_args()
require_compute()
assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
torch.set_num_threads(2)
root=scratch_root(); cfg=config()
people=json.loads((ROOT/"configs/splits/stage_a_v1.json").read_text())["subjects"]
common=json.loads((ROOT/"configs/stage_a_channels.json").read_text())["readout_common"]
assert sha256(ROOT/cfg["model"]["weights"]) == cfg["model"]["weights_sha256"]
model=load_backbone(ROOT/cfg["model"]["weights"], k=4).cuda().eval()
receipt=provenance(); receipt.update(gpu=torch.cuda.get_device_name(),worker=args.worker,subjects=[])
start=time.monotonic()

@torch.inference_mode()
def extract(x, picks, prefix_path=None, whitening=None):
    features=[]
    prefix_array=None
    for first in range(0,len(x),32):
        batch=np.array(x[first:first+32],dtype=np.float32,copy=True)
        if whitening is not None:
            batch=np.asarray(whitening @ batch,dtype=np.float32)
        tensor=torch.as_tensor(batch,device="cuda")
        prefix=model.prefix(tensor)
        if prefix_path is not None:
            if prefix_array is None:
                prefix_array=np.lib.format.open_memmap(prefix_path,mode="w+",dtype=np.float16,shape=(len(x),*prefix.shape[1:]))
            prefix_array[first:first+len(batch)] = prefix.cpu().numpy().astype(np.float16)
            # B0 uses exactly the quantized prefix that later stages will read.
            prefix=prefix.half().float()
        full=model.suffix(prefix)
        pooled=full[:,picks].mean(2).flatten(1)
        assert torch.isfinite(pooled).all()
        features.append(pooled.cpu().numpy().astype(np.float32))
    if prefix_array is not None:prefix_array.flush()
    return np.concatenate(features)

selected=people[args.worker::args.workers]
if args.limit:selected=selected[:args.limit]
for dataset,subject in selected:
    src=root/"processed"/dataset/f"sub-{subject:03d}"
    out=root/"features_v2"/dataset/f"sub-{subject:03d}"
    if (out/"done.json").exists():
        receipt["subjects"].append(json.loads((out/"done.json").read_text()));continue
    out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    meta=json.loads((src/"metadata.json").read_text())
    picks=[meta["channels"].index(c) for c in common]
    signals={"tasks":np.load(src/"tasks.npy",mmap_mode="r")}
    signals.update({eye:np.load(src/f"rest_{eye}.npy",mmap_mode="r") for eye in meta["rest"]})
    b0=extract(signals["tasks"],picks,out/"prefix_tasks.npy")
    np.save(out/"b0.npy",b0)
    for eye in meta["rest"]:
        extract(signals[eye],picks,out/f"prefix_rest_{eye}.npy")
    refs=ea_references(signals)
    for name,w in refs.items():
        # Whitening removes physical scale; no second /100 is applied.
        np.save(out/f"ea_{name}.npy",extract(signals["tasks"],picks,whitening=w))
    for name,covs in classic_covariances(signals,picks).items():np.save(out/f"classic_{name}.npy",covs)
    np.save(out/"labels.npy",np.load(src/"labels.npy"))
    torch.cuda.synchronize()
    row={"dataset":dataset,"subject":subject,"seconds":time.monotonic()-began,
         "prefix_bytes":sum(p.stat().st_size for p in out.glob("prefix_*.npy")),
         "feature_bytes":sum(p.stat().st_size for p in out.glob("*.npy")),
         "peak_gpu_bytes":torch.cuda.max_memory_allocated(),"trials":len(b0),
         "git_commit":receipt["git_commit"],"gpu":receipt["gpu"],"common_readout_channels":common}
    save_json(out/"done.json",row);receipt["subjects"].append(row)
    save_json(root/f"receipts/cache-worker-{args.worker}.json",receipt)
    print(json.dumps(row),flush=True)
receipt.update(status="complete",elapsed_seconds=time.monotonic()-start)
save_json(root/f"receipts/cache-worker-{args.worker}.json",receipt)
