"""CPU-only preparation of the six preselected training/validation timing subjects."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.cross_model_data import prepare_labram, starter_people, timing_people
from subject_context.stage_a_common import require_compute, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subject-job", type=int, help="Formal CPU array: index into the frozen 235-person roster")
    args = parser.parse_args()
    require_compute()
    began = time.monotonic()
    if args.subject_job is not None:
        roster = starter_people()
        if not 0 <= args.subject_job < len(roster):
            raise ValueError("Subject array index outside the frozen roster")
        people = {"formal_preprocessing": [roster[args.subject_job]]}
    else:
        people = timing_people()
    rows = []
    for role, subjects in people.items():
        for dataset, subject in subjects:
            start = time.monotonic()
            row = prepare_labram(args.stage_a_root, args.out, dataset, subject)
            rows.append({**row, "role": role, "seconds": time.monotonic() - start})
            print(f"{role} {dataset} {subject}: prepared", flush=True)
    receipt = (args.out / "timing/preparation.json" if args.subject_job is None else
               args.out / "preparation" / f"subject-{args.subject_job:03d}.json")
    save_json(receipt, {"status": "complete", "people": rows,
              "seconds": time.monotonic() - began, "GPU_jobs": 0, "scientific_selection": False})


if __name__ == "__main__":
    main()
