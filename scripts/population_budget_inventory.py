"""Read-only inventory of existing checkpoints; no model evaluation or training."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


a = Path("/path/to/scratch/eeg-cross-subject/stage-a")
x = Path("/path/to/results/eeg-cross-subject/stage-x-20260927")
rows = []
for model in ["REVE", "LaBraM", "CBraMod"]:
    for job in range(25):
        fold, seed = job//5, [11,23,37,53,71][job%5]
        name = f"fold-{fold}_seed-{seed}"
        if model == "CBraMod":
            population = a / "stage_w/population/runs" / name
            diagnostic = a / "stage_w/diagnostic1/runs" / name
            selection = read(a / "stage_c/runs" / name / "selection.json")
            assert selection["base_head"] == read(a / "stage_b/runs" / name / "selection.json")["base_head"]
            c = read(population / "selection.json")["m_lora"]
            hp = c["hyperparameters"]
            baseline = c["original_steps"] + c["selected_additional_step"]
            checkpoint = population / "model-m_lora-main.pt"
            expected = c["G_sha256"]
            head = a / "baselines_v2" / ("B0_"+selection["base_head"]) / name / "head.pt"
            personal = diagnostic / "personal-lora8.pt"
            personal_selection = diagnostic / "selection.json"
            personal_hp = read(personal_selection)["lora8"]
            head_kind = selection["base_head"]
            timing = {"old_W_population_seconds_both_families": read(population/"run.json")["seconds"],
                      "old_W_diagnostic_seconds_both_families": read(diagnostic/"run.json")["seconds"],
                      "old_W_lora_additional_steps": c["additional_steps_run"]}
        else:
            source = x / model / "runs" / name
            population, diagnostic = source / "G", source / "diagnostic"
            c = read(population / "convergence.json")
            hp = read(population / "initial_selection.json")
            baseline = hp["episodes"] + c["selected_additional_step"]
            checkpoint = population / "continuation/selected.pt"
            locked = read(source / "frozen_before_test.json")
            expected = locked["files"]["G/continuation/selected.pt"]
            head, personal = source / "B0/selected.pt", diagnostic / "personal-full.pt"
            personal_selection = source / "personal/selection.json"
            personal_hp = read(personal_selection)["full"]
            head_kind = read(source / "B0/selection.json")["kind"]
            timing = {"old_G_seconds_all_initial_candidates_and_continuation": c["seconds"],
                      "old_G_updates_all_candidates_and_continuation": 6000+c["additional_steps_run"]}
        assert baseline > 0 and baseline % 250 == 0
        source_sha = digest(checkpoint)
        assert source_sha == expected
        assert read(diagnostic / "run.json")["status"] == "complete" if model == "CBraMod" else True
        paths = {"checkpoint": checkpoint, "B0_head": head, "personal_parameters": personal,
                 "personal_selection": personal_selection}
        rows.append({"array_index":len(rows),"model":model,"job":job,"fold":fold,"seed":seed,
                     "baseline_total_steps":baseline,"target_total_steps":{"1x":baseline,"2x":2*baseline,"4x":4*baseline},
                     "maximum_added_steps":3*baseline,"population_hyperparameters":hp,"head_kind":head_kind,
                     "personal_hyperparameters_1x":personal_hp,"source_population":str(population),
                     "source_diagnostic":str(diagnostic),
                     "files":{k:{"path":str(p),"bytes":p.stat().st_size,"sha256":source_sha if k=="checkpoint" else digest(p)} for k,p in paths.items()},
                     "optimizer_resume":model!="CBraMod","historical_timing":timing})
original = Path("/path/to/eeg-cross-subject/stage-w/configs/splits")
current = Path("/path/to/eeg-cross-subject/stage-x/configs/splits")
for filename in ["stage_a_v1.json", "stage_b_halves_v1.json"]:
    assert read(original/filename) == read(current/filename), "Historical split semantic change"
result = {"scope":"existing_artifacts_only_no_new_performance", "runs":rows,
          "total_maximum_new_updates":sum(r["maximum_added_steps"] for r in rows),
          "historical_and_current_split_semantics_equal":True}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"runs":len(rows),"total_maximum_new_updates":result["total_maximum_new_updates"],"all_checkpoint_hashes_verified":True}))
