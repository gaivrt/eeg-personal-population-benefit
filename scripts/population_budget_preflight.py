"""CPU-only source compatibility check. No EEG loading, model forward or GPU job."""
import json
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model import load_backbone, PopulationModel, PersonalModel
from subject_context.model import load_backbone as load_cb
from subject_context.stage_a_common import ROOT, sha256
from subject_context.stage_b import Readout
from subject_context.stage_c import ContextModel, config_c
from subject_context.stage_c2 import PersonalModel as CBPersonal
from subject_context.stage_c_train import load_snapshot

torch.set_num_threads(4)
inventory_path = ROOT / "docs/population_budget_engineering_2026-09-27/source_inventory.json"
inventory = json.loads(inventory_path.read_text())
results = []
for name in ["REVE", "LaBraM", "CBraMod"]:
    source = next(r for r in inventory["runs"] if r["model"] == name)
    head = Readout(torch.load(source["files"]["B0_head"]["path"], map_location="cpu", weights_only=False), source["head_kind"])
    if name == "CBraMod":
        backbone = load_cb('/path/to/eeg-cross-subject/stage-w/weights/cbramod.pth', k=4)
        population = ContextModel(backbone, head, "m_lora", 5400, 756, config_c())
    else:
        backbone = load_backbone(name, '/path/to/eeg-cross-subject/stage-x/weights/cross_model')
        population = PopulationModel(backbone, head)
    state = torch.load(source["files"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    load_snapshot(population, state if name == "CBraMod" else state["parameters"])
    if name != "CBraMod":
        assert state["step"] + source["population_hyperparameters"]["episodes"] == source["baseline_total_steps"]
        assert all(k in state for k in ["optimizer", "numpy_rng", "torch_generator", "torch_rng", "cuda_rng"])
    personal = CBPersonal(population, "lora8", 8) if name == "CBraMod" else PersonalModel(population)
    params = torch.load(source["files"]["personal_parameters"]["path"], map_location="cpu", weights_only=True)
    for values in params.values():
        personal.load_personal(values)
    results.append({"model": name, "group_trainable_tensors": len(population.trainable()),
                    "personal_parameter_tensors": len(personal.trainable()), "historical_personal_people_loaded": len(params),
                    "source_shapes_match": True})
    del personal, population, backbone, head, state, params
print(json.dumps({"scope": "CPU_only_no_EEG_no_forward_no_GPU", "inventory_sha256": sha256(inventory_path), "models": results}))
