"""Retain immutable terminal-run artifacts in home and export non-checkpoint files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["REVE", "LaBraM", "CBraMod"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not args.tag.replace("-", "").isalnum():
        raise ValueError("CPU Slurm allocation and a simple archive tag required")
    source_root = Path('/path/to/scratch/eeg-cross-subject/population-budget-20260927')
    retained = Path('/path/to/results/eeg-cross-subject/population-budget-20260927')
    inventory = json.loads((ROOT / 'docs/population_budget_engineering_2026-09-27/source_inventory.json').read_text())['runs']
    receipt = source_root / f'retention-{args.tag}.json'
    if receipt.exists():
        raise FileExistsError("Do not replace an earlier retention receipt")
    terminal, observed = [], {}
    for src in inventory:
        run_dir = source_root / src['model'] / 'runs' / f"fold-{src['fold']}_seed-{src['seed']}"
        run = json.loads((run_dir / 'run.json').read_text()) if (run_dir / 'run.json').exists() else {'status': 'pending'}
        observed.setdefault(src['model'], []).append(run['status'])
        if src['model'] in args.models or (args.include_failed and run['status'] == 'failed'):
            if run['status'] not in ('complete', 'failed'):
                raise ValueError(f"Requested model has unfinished trajectory: {run_dir}")
            terminal.append((src, run_dir, run['status']))
    for model in args.models:
        if len(observed.get(model, [])) != 25:
            raise ValueError("Requested model does not have 25 inventory trajectories")
    files = {}
    for src, run_dir, status in terminal:
        for file in run_dir.rglob('*'):
            if file.is_file():
                files[file.relative_to(source_root).as_posix()] = file
        log = source_root / 'logs' / f"curve-1911397_{src['array_index']}.out"
        files[log.relative_to(source_root).as_posix()] = log
        for name, record in src['files'].items():
            file = Path(record['path'])
            if digest(file) != record['sha256']:
                raise ValueError("Historical source changed before retention")
            files[f"sources/{src['model']}/fold-{src['fold']}_seed-{src['seed']}/{name}{file.suffix}"] = file
    for file in (source_root / 'report').rglob('*'):
        if file.is_file():
            files[f"audit_snapshots/{args.tag}/report/{file.relative_to(source_root / 'report').as_posix()}"] = file
    for pattern in ('*-receipt.json', 'integrity-audit-*.json', 'submission_status.json'):
        for file in source_root.glob(pattern):
            files[f"audit_snapshots/{args.tag}/{file.name}"] = file
    for file in (source_root / 'logs').glob('*audit*.out'):
        files[f"audit_snapshots/{args.tag}/logs/{file.name}"] = file
    for folder in ('src', 'configs', 'scripts'):
        for file in (ROOT / folder).rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                files[f"audit_snapshots/{args.tag}/execution_code/{file.relative_to(ROOT).as_posix()}"] = file
    for relative in ('authorization.json', 'docs/population_budget_protocol.md',
                     'docs/population_budget_engineering_2026-09-27/source_inventory.json'):
        files[f"audit_snapshots/{args.tag}/execution_code/{relative}"] = ROOT / relative
    retained.mkdir(parents=True, exist_ok=True)
    needed = sum(p.stat().st_size for name, p in files.items() if not (retained / name).exists())
    if shutil.disk_usage(retained).free < needed + 1024**3:
        raise OSError("Insufficient retained storage")
    manifest = []
    for name, original in sorted(files.items()):
        target = retained / name
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(original)
        if not target.exists():
            shutil.copy2(original, target)
        if digest(target) != sha:
            raise ValueError(f"Retained copy differs; preserve both and inspect: {name}")
        manifest.append({'path': name, 'bytes': original.stat().st_size, 'sha256': sha,
                         'source': str(original), 'non_checkpoint': '.pt' not in original.suffixes})
    archive_dir = source_root / 'archives'
    archive_dir.mkdir(exist_ok=True)
    archive = archive_dir / f'noncheckpoint-{args.tag}.tar.gz'
    if archive.exists():
        raise FileExistsError("Preserve existing export archive")
    with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
        for item in manifest:
            if item['non_checkpoint']:
                bundle.add(retained / item['path'], arcname=item['path'], recursive=False)
    payload = {'recorded_at_UTC': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
               'models': args.models, 'terminal_runs': [{k: src[k] for k in ('model', 'fold', 'seed', 'array_index')} | {'status': status} for src, _, status in terminal],
               'retained_directory': str(retained), 'files': manifest, 'file_count': len(manifest),
               'retained_bytes_in_this_manifest': sum(r['bytes'] for r in manifest),
               'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive)}
    receipt.write_text(json.dumps(payload, indent=2) + '\n')
    shutil.copy2(receipt, retained / 'audit_snapshots' / args.tag / 'retention_manifest.json')
    print(json.dumps({k: v for k, v in payload.items() if k not in ('files', 'terminal_runs')}))


if __name__ == '__main__':
    main()
