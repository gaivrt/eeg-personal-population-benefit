"""Lee session 1 external replication: fixed starter hyperparameters, no grid search."""
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
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.data import assert_split
from subject_context.model import load_backbone, state_hash
from subject_context.stage_a_common import ROOT, config, provenance, require_compute, save_json, scratch_root, sha256
from subject_context.stage_b import Readout, ba, temporal_halves
from subject_context.stage_c import ContextModel, backbone_features, config_c
from subject_context.stage_c_context import TangentFeatures, band_covariances
from subject_context.stage_c_train import snapshot, train
from subject_context.stage_c2 import PersonalModel, fit, predict
from subject_context.stage_c3 import nearest
from subject_context.stage_w import config_w, continuation


def fixed_head(groups, kind, hp, seed):
    """The original Lee head training, with an externally frozen epoch/lr choice."""
    arrays = torch.cat([s["features"] for s in groups["train"]]).cpu().numpy()
    mean, std = arrays.mean(0, dtype=np.float64), np.maximum(arrays.std(0, dtype=np.float64), 1e-6)
    features = torch.as_tensor((arrays-mean)/std, device="cuda", dtype=torch.float32)
    labels = torch.cat([s["labels"] for s in groups["train"]])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    head = (nn.Linear(len(mean), 2) if kind == "linear" else nn.Sequential(
        nn.Linear(len(mean), 128), nn.GELU(), nn.Linear(128, 2))).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=hp["learning_rate"], weight_decay=.01)
    for epoch in range(1, int(hp["epoch"])+1):
        head.train()
        order = torch.randperm(len(labels), device="cuda")
        for take in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features[take]), labels[take])
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        head.eval()
    return {"head": {k:v.cpu() for k,v in head.state_dict().items()}, "feature_mean": mean,
            "feature_std": std, "selection": hp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=int, required=True, choices=range(25))
    parser.add_argument("--bnci-followup", action="store_true", help="Approved BNCI session-2 replication; frozen starter settings")
    args = parser.parse_args()
    require_compute()
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    root, c_cfg, w_cfg = scratch_root(), config_c(), config_w()
    config_path = ROOT / "configs/stage_w.yaml"
    prefix = "stage_w/lee"
    if args.bnci_followup:
        config_path = ROOT / "configs/stage_w_followup.yaml"
        followup = yaml.safe_load(config_path.read_text())
        w_cfg["external"] = followup["bnci"]
        prefix = "stage_w_followup/bnci"
    external = w_cfg["external"]
    fold_index, seed = args.job // 5, w_cfg["seeds"][args.job % 5]
    name = f"fold-{fold_index}_seed-{seed}"
    out = root / prefix / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.glob("model-*.pt")) and not (out / "run.json").exists(), "Do not overwrite an existing attempt"
    prep = root / prefix
    prepared = json.loads((prep / "prepared.json").read_text())
    assert prepared["status"] == "complete" and prepared["subjects"] == external["subjects"]
    assert prepared["w_config_sha256"] == sha256(config_path)
    if prepared["rest_available_subjects"] != external["subjects"]:
        raise RuntimeError("Original C training requires rest; retain the audit and report unavailable subjects before any protocol change")
    fold = json.loads((prep / "splits.json").read_text())["folds"][fold_index]
    split = {r: [tuple(k) for k in fold[r]] for r in ["train", "validation", "test"]}
    assert_split(*split.values())
    halves = {(r["dataset"], r["subject"]): r for r in json.loads((prep / "halves.json").read_text())["subjects"]}
    source_c = root / "stage_c/runs" / name
    hp_c = json.loads((source_c / "selection.json").read_text())
    source_d1 = root / "stage_w/diagnostic1/runs" / name
    source_fs = root / "stage_w/diagnostic3/runs" / name
    assert json.loads((source_d1 / "run.json").read_text())["status"] == "complete"
    assert json.loads((source_fs / "run.json").read_text())["status"] == "complete"
    hp_d1 = json.loads((source_d1 / "selection.json").read_text())
    hp_fs = json.loads((source_fs / "selection.json").read_text())
    kind = hp_c["base_head"]
    source_b0 = root / "baselines_v2" / f"B0_{kind}" / name
    hp_head = json.loads((source_b0 / "run.json").read_text())["selection"]
    start = time.monotonic()
    receipt = provenance()
    receipt.update(fold=fold_index, seed=seed, dataset=external["dataset"], session=external["session"],
        w_config_sha256=sha256(config_path), split_sha256=sha256(prep / "splits.json"),
        halves_sha256=sha256(prep / "halves.json"), gpu=torch.cuda.get_device_name(),
        head_kind=kind, head_hyperparameters=hp_head, population_hyperparameters=hp_c,
        personal_hyperparameters=hp_d1, few_shot_hyperparameters=hp_fs, lee_hyperparameter_search=False,
        source_config_hashes={str(p.relative_to(root)): sha256(p) for p in [source_c / "selection.json",
            source_d1 / "selection.json", source_fs / "selection.json", source_b0 / "run.json"]})
    if args.bnci_followup:
        linear_source = root / "baselines_v2/B0_linear" / name / "run.json"
        hp_linear = json.loads(linear_source.read_text())["selection"]
        receipt.update(external_hyperparameter_search=False, primary_reference="G",
                       linear_probe_hyperparameters=hp_linear, no_donor_policy="unavailable_not_imputed")
        receipt["source_config_hashes"][str(linear_source.relative_to(root))] = sha256(linear_source)
    save_json(out / "frozen_hyperparameters.json", receipt)
    backbone = load_backbone(ROOT / config()["model"]["weights"], k=4).cuda().eval()
    frozen = state_hash(backbone)
    common = (prepared["readout_channels"] if args.bnci_followup else
              [c for c in json.loads((ROOT / "configs/stage_a_channels.json").read_text())["readout_common"] if c != "FCz"])
    assert len(common) == (19 if args.bnci_followup else 26)

    @torch.no_grad()
    def pooled(x, picks):
        return torch.cat([backbone_features(backbone.model, part.float(), picks) for part in x.split(32)])

    def load_subject(key):
        dataset, subject = key
        directory = root / "processed" / dataset / f"sub-{subject:03d}"
        record = halves[key]
        assert sha256(directory / "trials.csv") == record["trial_table_sha256"]
        table = pd.read_csv(directory / "trials.csv")
        fit_rows, query, counts = temporal_halves(table)
        assert fit_rows.tolist() == record["fit_indices"] and query.tolist() == record["query_indices"]
        meta = json.loads((directory / "metadata.json").read_text())
        picks = [meta["channels"].index(c) for c in common]
        x = torch.as_tensor(np.load(directory / "tasks.npy"), device="cuda")
        y = np.load(directory / "labels.npy")
        np.testing.assert_array_equal(y, table.label)
        rest = np.load(directory / "rest_open.npy")[np.load(directory / "rest_open_valid.npy")]
        assert len(rest)*4 >= 30 and meta["rest"]["open"]["protocol_order"] < table.protocol_order.min()
        return {"key": key, "eeg": x, "picks": picks, "labels": torch.tensor(y, device="cuda"),
            "fit": torch.tensor(fit_rows, device="cuda"), "query": torch.tensor(query, device="cuda"),
            "query_labels": y[query], "trial_ids": table.trial_id.tolist(), "counts": counts,
            "features": pooled(x, picks), "rest_pooled": pooled(torch.as_tensor(rest, device="cuda"), picks),
            "bands": band_covariances(rest[:, picks].astype(np.float64), c_cfg["context"]["bands_hz"], c_cfg["context"]["filter_order"])}

    groups = {role: [load_subject(key) for key in split[role]] for role in ["train", "validation"]}
    # B0 is trained for the exact number of epochs chosen on the starter fold.
    # Its normalization is fitted only on Lee training subjects.
    state = fixed_head(groups, kind, hp_head, seed)
    torch.save(state, out / "B0-head.pt")
    readout = Readout(state, kind).cuda()
    linear_readout = None
    if args.bnci_followup:
        linear_state = state if kind == "linear" else fixed_head(groups, "linear", hp_linear, seed)
        torch.save(linear_state, out / "B0-linear-head.pt")
        linear_readout = Readout(linear_state, "linear").cuda()
    tangent = TangentFeatures().fit([s["bands"] for s in groups["train"]])
    for group in groups.values():
        for s in group:
            s["emb"] = readout.normalize(s["rest_pooled"])
            s["tan"] = torch.as_tensor(tangent(s.pop("bands")), device="cuda")
    models, statuses, curve, loss_rows, trace = {}, {}, [], [], []
    for method in ["m_film", "m_lora"]:
        hp = hp_c[f"{method}_main"]
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = ContextModel(backbone, readout, method, len(common)*200, len(common)*(len(common)+1), c_cfg).cuda()
        initial_cfg = copy.deepcopy(c_cfg)
        initial_cfg["training"]["episodes"] = [int(hp["episodes"])]
        train(model, groups["train"], hp, initial_cfg, seed,
              lambda step: torch.save(snapshot(model), out / f"B3-{method}.pt"),
              on_step=lambda step,key,ce,obj: trace.append({"method":method,"phase":"initial", "step":step,
                  "dataset":key[0],"subject":key[1],"ce":ce,"objective":obj}))
        curves, losses, episodes, statuses[method] = continuation(model, groups["train"], groups["validation"],
            hp, c_cfg, w_cfg, seed, lambda state,step: torch.save(state, out / f"model-{method}-main.pt"))
        curve.extend({"method":method,**r} for r in curves)
        loss_rows.extend({"method":method,**r} for r in losses)
        trace.extend({"method":method,"phase":"continuation",**r} for r in episodes)
        models[method] = model
        print(json.dumps({"method":method,**statuses[method]}), flush=True)
    pd.DataFrame(curve).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(out / "subject_losses.csv", index=False)
    pd.DataFrame(trace).to_csv(out / "training_steps.csv", index=False)
    save_json(out / "G_selection.json", statuses)
    # Test data first loaded after all checkpoint choices are final.
    tests = [load_subject(key) for key in split["test"]]
    rest_features = {}
    for s in tests:
        embedding = readout.normalize(s["rest_pooled"]).mean(0).cpu().numpy()
        tan = tangent(s.pop("bands")).mean(0)
        rest_features[s["key"]] = np.concatenate([embedding/np.sqrt(len(embedding)), tan/np.sqrt(len(tan))])
    np.savez(out / "rest_features.npz", **{f"{d}:{n}":v for (d,n),v in rest_features.items()})
    predictions, baseline, d1_rows, nn_rows, few_rows, resources = [], [], [], [], [], []

    def record(s, condition, pred, shots=0, donor=None):
        predictions.extend({"dataset":s["key"][0],"subject":s["key"][1],"condition":condition,
            "shots":shots,"donor":donor,"trial_id":s["trial_ids"][i],"label":int(s["query_labels"][j]),
            "prediction":int(pred[j])} for j,i in enumerate(s["query"].cpu().tolist()))

    rng = np.random.default_rng(seed)
    draws = {}
    for s in tests:
        others = [t["key"] for t in tests if t["key"] != s["key"]]
        draws[s["key"]] = [others[i] for i in rng.integers(len(others), size=10)] if others else []
        with torch.no_grad():
            pred = readout(s["features"]).argmax(1).cpu().numpy()[s["query"].cpu().numpy()]
        record(s, "B0", pred)
        baseline.append({"dataset":s["key"][0],"subject":s["key"][1],"condition":"B0","ba":ba(s["query_labels"],pred)})
        if linear_readout is not None:
            with torch.no_grad():
                linear_pred = linear_readout(s["features"]).argmax(1).cpu().numpy()[s["query"].cpu().numpy()]
            record(s, "B0_linear", linear_pred)
            baseline.append({"dataset":s["key"][0],"subject":s["key"][1],"condition":"B0_linear",
                             "ba":ba(s["query_labels"],linear_pred)})
    for method, variant in [("m_film", "film_offset"), ("m_lora", "lora8")]:
        # Exactly the C2 default-z route; then the original personal parameter class.
        reference = PersonalModel(models[method], "film_offset" if method == "m_film" else "mix_offset")
        g_ba = {}
        for s in tests:
            all_idx = torch.arange(len(s["labels"]), device="cuda")
            pred = predict(reference, s, all_idx)[s["query"].cpu().numpy()]
            g_ba[s["key"]] = ba(s["query_labels"], pred)
            record(s, "G_"+variant, pred)
            baseline.append({"dataset":s["key"][0],"subject":s["key"][1],"condition":"G_"+variant,"ba":g_ba[s["key"]]})
        del reference
        model = PersonalModel(models[method], variant, 8)
        hp = hp_d1[variant]
        params, own, matrix = {}, {}, {}
        for s in tests:
            result, params[s["key"]], resource = fit(model,s,hp["learning_rate"],hp["weight_decay"],[hp["steps"]],seed+s["key"][1])
            own[s["key"]] = result[hp["steps"]]["ba"]
            resources.append({"variant":variant,"shots":"all",**resource})
        torch.save({f"{d}:{n}": [v.cpu() for v in p] for (d,n),p in params.items()},out / f"personal-{variant}.pt")
        for s in tests:
            for donor in tests:
                model.load_personal(params[donor["key"]])
                pred = predict(model,s,s["query"])
                matrix[(s["key"],donor["key"])] = ba(s["query_labels"],pred)
                record(s,"swap_"+variant,pred,donor=donor["key"][1])
            assert abs(matrix[(s["key"],s["key"])]-own[s["key"]]) < 1e-12
            swap = float(np.mean([matrix[(s["key"],d)] for d in draws[s["key"]]])) if draws[s["key"]] else float("nan")
            d1_rows.append({"variant":variant,"dataset":s["key"][0],"subject":s["key"][1],"fold":fold_index,"seed":seed,
                "b3_ba":g_ba[s["key"]],"own_ba":own[s["key"]],"swap_ba":swap,"upper_bound":own[s["key"]]-swap,
                "own_gain":own[s["key"]]-g_ba[s["key"]],"swap_gain":swap-g_ba[s["key"]],
                "swap_donors":" ".join(str(d[1]) for d in draws[s["key"]]),
                "swap_available":bool(draws[s["key"]]), **s["counts"]})
            others = [t["key"] for t in tests if t["key"] != s["key"]]
            if not others:
                assert args.bnci_followup
                nn_rows.append({"variant":variant,"source":"rest","k":1,"dataset":s["key"][0],"subject":s["key"][1],
                    "fold":fold_index,"seed":seed,"neighbours":"","nearest_ba":float("nan"),"random_ba":float("nan"),
                    "difference":float("nan"),"available":False,"reason":"no_same_fold_test_donor"})
                continue
            donor = nearest(rest_features,s["key"],others,1)[0]
            nn_rows.append({"variant":variant,"source":"rest","k":1,"dataset":s["key"][0],"subject":s["key"][1],
                "fold":fold_index,"seed":seed,"neighbours":str(donor[1]),"nearest_ba":matrix[(s["key"],donor)],
                "random_ba":swap,"difference":matrix[(s["key"],donor)]-swap,
                "available":True,"reason":"single_donor_nearest_equals_random" if len(others)==1 else ""})
        if variant == "lora8":
            for n in w_cfg["external"]["shots"]:
                hp = hp_fs[f"lora8_n{n}"]
                for s in tests:
                    view = {**s,"fit":s["fit"][:n]}
                    result,_,resource = fit(model,view,hp["learning_rate"],hp["weight_decay"],[hp["steps"]],seed+s["key"][1])
                    r = result[hp["steps"]]
                    record(s,"few_lora8",r["predictions"],shots=n)
                    # Same C3 second-half batch route for the paired n=0 baseline.
                    zero = PersonalModel(models[method],"mix_offset")
                    zero_pred = predict(zero,s,s["query"])
                    g_query = ba(s["query_labels"],zero_pred)
                    record(s,"few_G",zero_pred,shots=n)
                    del zero
                    few_rows.append({"variant":variant,"shots":n,"dataset":s["key"][0],"subject":s["key"][1],
                        "fold":fold_index,"seed":seed,"b3_ba":g_query,"ba":r["ba"],"gain":r["ba"]-g_query,
                        "full_gain":own[s["key"]]-g_ba[s["key"]],
                        "one_class_only":len(set(s["labels"][s["fit"][:n]].cpu().tolist()))<2})
                    resources.append({"variant":variant,"shots":n,**resource})
        del model
    for filename,rows in [("baseline.csv",baseline),("subjects.csv",d1_rows),("neighbours.csv",nn_rows),
                          ("few_shot.csv",few_rows),("fit_resources.csv",resources),("predictions.csv.gz",predictions)]:
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    assert state_hash(backbone) == frozen
    receipt.update(status="complete",seconds=time.monotonic()-start,families=statuses,
        backbone_unchanged=True,readout_channels=common,start_label="G",peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save_json(out / "run.json",receipt)


if __name__ == "__main__":
    main()
