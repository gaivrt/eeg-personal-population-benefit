"""Recompute saved metrics and check each query against the frozen temporal split."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from subject_context.stage_a_common import ROOT, save_json, scratch_root, sha256
from subject_context.stage_b import ba, config_b

root=scratch_root();cfg=config_b()
folds=json.loads((ROOT/cfg["split_file"]).read_text())["folds"]
halves=json.loads((ROOT/"configs/splits/stage_b_halves_v1.json").read_text())["subjects"]
expected={}
for h in halves:
    t=pd.read_csv(root/"processed"/h["dataset"]/f'sub-{h["subject"]:03d}'/"trials.csv")
    expected[(h["dataset"],h["subject"])]=t.iloc[h["query_indices"]].set_index("trial_id").label.sort_index()
count=0;predictions=0
for d in sorted((root/"stage_b/runs").glob("fold-*_seed-*")):
    meta=json.loads((d/"run.json").read_text())
    assert meta["stage_b_config_sha256"]==sha256(ROOT/"configs/stage_b.yaml")
    assert meta["split_sha256"]==sha256(ROOT/cfg["split_file"])
    s=pd.read_csv(d/"subjects.csv");p=pd.read_csv(d/"predictions.csv")
    original=pd.read_csv(root/"baselines_v2"/f'B0_{meta["base_head"]}'/
                         f'fold-{meta["fold"]}_seed-{meta["seed"]}'/"predictions.csv").set_index("trial_id")
    baseline=p.drop_duplicates("trial_id").set_index("trial_id").sort_index()
    np.testing.assert_array_equal(baseline.b0_prediction,original.loc[baseline.index].prediction)
    assert set(map(tuple,s[["dataset","subject"]].drop_duplicates().to_numpy()))==set(map(tuple,folds[meta["fold"]]["test"]))
    assert not p.duplicated(["variant","trial_id"]).any()
    for row in s.itertuples():
        g=p[(p.variant==row.variant)&(p.dataset==row.dataset)&(p.subject==row.subject)]
        pd.testing.assert_series_equal(g.set_index("trial_id").label.sort_index(),expected[(row.dataset,row.subject)])
        np.testing.assert_allclose([ba(g.label,g.b0_prediction),ba(g.label,g.prediction)],
                                   [row.b0_ba,row.ba],atol=1e-14,rtol=0)
        assert abs(row.ba-row.b0_ba-row.gain)<1e-14
        count+=1;predictions+=len(g)
assert count==235*5*4
save_json(root/"stage_b/report/result_verification.json",{"status":"pass","subject_seed_variant_rows":count,
    "trial_prediction_rows":predictions,"checks":["all query trials equal the frozen second half",
    "all query labels equal saved source labels","all outer test subjects match the frozen fold",
    "recomputed baseline and adapted BA match every subject row","config and split hashes match",
    "B0 query predictions equal the corresponding Stage A checkpoint predictions"]})
print(json.dumps({"status":"pass","subject_seed_variant_rows":count,"trial_prediction_rows":predictions}))
