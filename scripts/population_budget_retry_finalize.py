"""CPU-only audit, explicit one-slot merge and retention after the single retry."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

CODE = Path(__file__).resolve().parents[1]
OUT = Path('/path/to/scratch/eeg-cross-subject/population-budget-retry1-20260927')
PRIOR = Path('/path/to/results/eeg-cross-subject/population-budget-20260927')
HOME = Path('/path/to/results/eeg-cross-subject/population-budget-retry1-20260927')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (OUT/'retention.json').exists(), 'Do not repeat completed finalization'
    auth = json.loads((CODE/'authorization.json').read_text())
    inventory = CODE/'docs/population_budget_engineering_2026-09-27/source_inventory.json'
    sources = json.loads(inventory.read_text())['runs']
    original_manifest = PRIOR/'audit_snapshots/final/retention_manifest.json'
    plan = json.loads((CODE/'finalization_plan.json').read_text())
    assert digest(original_manifest) == plan['prior_retention_manifest_sha256']
    prior = json.loads(original_manifest.read_text())
    # Verify the exact previously retained data before taking any of its 74 runs.
    for record in prior['files']:
        path = PRIOR/record['path']
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], record['path']
    rerun = OUT/'CBraMod/runs/fold-1_seed-37'
    run = json.loads((rerun/'run.json').read_text())
    assert run['status'] in ('complete','failed') and run['retry_attempt'] == 1
    assert run['authorization_sha256'] == digest(CODE/'authorization.json')
    subprocess.run([sys.executable, str(CODE/'scripts/population_budget_integrity.py'),
        '--out', str(OUT), '--inventory', str(inventory), '--report', str(OUT/'retry-integrity.json')], check=True)
    audit = json.loads((OUT/'retry-integrity.json').read_text())
    assert len(audit['runs']) == 1 and not audit['errors']
    merged = OUT/'merged'
    merged.mkdir(exist_ok=False)
    selection = []
    for src in sources:
        name = f"{src['model']}/runs/fold-{src['fold']}_seed-{src['seed']}"
        path = rerun if src['array_index'] == 57 else PRIOR/name
        state = json.loads((path/'run.json').read_text())['status']
        if src['array_index'] != 57:
            assert state == 'complete'
        target = merged/name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.symlink_to(path, target_is_directory=True)
        selection.append({k:src[k] for k in ('array_index','model','fold','seed')} |
                         {'source_directory':str(path),'status':state,'run_sha256':digest(path/'run.json'),
                          'retry_precision_exception':src['array_index']==57})
    (OUT/'merge_manifest.json').write_text(json.dumps({'runs':selection,'prior_retention_manifest_sha256':digest(original_manifest),
        'replaced_failed_run':auth['failed_run'],'retry_status':run['status'],
        'aggregation':'five-seed mean per subject, then median of original 235 subjects; missing remains NA'},indent=2)+'\n')
    subprocess.run([sys.executable,str(CODE/'scripts/population_budget_results.py'),'--out',str(merged),
        '--inventory',str(inventory),'--stage-a-root','/path/to/scratch/eeg-cross-subject/stage-a'],check=True)
    submit = json.loads((OUT/'submission_receipt.json').read_text())
    accounting = subprocess.check_output(['sacct','-X','-j',submit['job_id'],
        '--format=JobID%30,State,ElapsedRaw,AllocTRES%150,ExitCode,Start,End','-P'],text=True)
    (OUT/'retry_gpu_accounting.psv').write_text(accounting)
    # Retain only new data here; the prior 74 trajectories remain in their verified archive.
    files = {}
    for folder in ('CBraMod','logs'):
        for path in (OUT/folder).rglob('*'):
            if path.is_file() and not path.name.startswith('finalize-'):
                files[path.relative_to(OUT).as_posix()] = path
    for path in (merged/'report').rglob('*'):
        if path.is_file():
            files['report/'+path.relative_to(merged/'report').as_posix()] = path
    for path in OUT.glob('*.json'):
        files[path.name] = path
    files['retry_gpu_accounting.psv'] = OUT/'retry_gpu_accounting.psv'
    files['prior_retention_manifest.json'] = original_manifest
    for folder in ('src','configs','scripts'):
        for path in (CODE/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                files['execution_code/'+path.relative_to(CODE).as_posix()] = path
    for rel in ('authorization.json','deployment_manifest.json','finalization_plan.json','docs/population_budget_protocol.md',
                'docs/population_budget_retry_protocol.md','docs/population_budget_engineering_2026-09-27/source_inventory.json'):
        files['execution_code/'+rel] = CODE/rel
    HOME.mkdir(exist_ok=False)
    assert shutil.disk_usage(HOME).free > sum(p.stat().st_size for p in files.values()) + 1024**3
    manifest = []
    for name,source in sorted(files.items()):
        target = HOME/name
        target.parent.mkdir(parents=True,exist_ok=True)
        expected = digest(source)
        shutil.copy2(source,target)
        assert digest(target) == expected,name
        manifest.append({'path':name,'bytes':source.stat().st_size,'sha256':expected,
                         'source':str(source),'non_checkpoint':'.pt' not in source.suffixes})
    archive = OUT/'noncheckpoint-retry1.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
        for item in manifest:
            if item['non_checkpoint']:
                bundle.add(HOME/item['path'],arcname=item['path'],recursive=False)
    payload = {'recorded_at_UTC':datetime.now(timezone.utc).isoformat(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'retry_status':run['status'],'files':manifest,'file_count':len(manifest),
        'retained_bytes':sum(r['bytes'] for r in manifest),'retained_directory':str(HOME),
        'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),
        'prior_files_reverified':len(prior['files']),'experiments_frozen':True}
    (OUT/'retention.json').write_text(json.dumps(payload,indent=2)+'\n')
    shutil.copy2(OUT/'retention.json',HOME/'retention.json')
    print(json.dumps({k:v for k,v in payload.items() if k!='files'}),flush=True)


if __name__ == '__main__':
    main()
