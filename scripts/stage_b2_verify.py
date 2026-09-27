"""Audit every B2 prediction, fixed split, selected hyperparameter, and prior-head pair."""
import json
import argparse
import hashlib
import subprocess
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from subject_context.stage_a_common import ROOT, save_json, scratch_root, sha256
from subject_context.stage_b import ba, select_candidate
from subject_context.stage_b2 import config_b2

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path)
parser.add_argument('--processed-root', type=Path)
parser.add_argument('--baselines-root', type=Path)
args = parser.parse_args()
root = args.root if args.root else scratch_root(); cfg = config_b2()
processed = args.processed_root if args.processed_root else root / 'processed'
baselines = args.baselines_root if args.baselines_root else root / 'baselines_v2'


def source_hash(relative):
    # Remote Git files use LF; Windows working trees may use CRLF.
    return hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)).hexdigest()
folds = json.loads((ROOT / cfg['split_file']).read_text())['folds']
halves = json.loads((ROOT / cfg['halves_file']).read_text())['subjects']
expected = {}
for h in halves:
    t = pd.read_csv(processed / h['dataset'] / f'sub-{h["subject"]:03d}' / 'trials.csv')
    assert sha256(processed / h['dataset'] / f'sub-{h["subject"]:03d}' / 'trials.csv') == h['trial_table_sha256']
    expected[(h['dataset'], h['subject'])] = t.iloc[h['query_indices']].set_index('trial_id').label.sort_index()
count = 0; predictions = 0; runs = 0
for d in sorted((root / 'stage_b2/runs').glob('fold-*_seed-*')):
    meta = json.loads((d / 'run.json').read_text()); runs += 1
    assert meta['status'] == 'complete'
    assert meta['stage_b2_config_sha256'] == source_hash('configs/stage_b2.yaml')
    assert meta['split_sha256'] == source_hash(cfg['split_file'])
    assert meta['halves_sha256'] == source_hash(cfg['halves_file'])
    assert not meta['activation_cache_used'] and meta['backbone_unchanged'] and meta['base_head_unchanged']
    s = pd.read_csv(d / 'subjects.csv'); p = pd.read_csv(d / 'predictions.csv')
    v = pd.read_csv(d / 'validation_search.csv', float_precision='round_trip')
    selection = json.loads((d / 'selection.json').read_text())
    val_set = set(map(tuple, folds[meta['fold']]['validation']))
    assert set(map(tuple, v[['dataset', 'subject']].drop_duplicates().to_numpy())) == val_set
    assert len(v) == len(val_set) * len(cfg['variants']) * 2 * 2 * 2
    for variant in cfg['variants']:
        actual = selection['parameters'][variant]
        selected = select_candidate(v[v.variant == variant].to_dict('records'))
        for key in ['steps', 'learning_rate', 'weight_decay']:
            assert actual[key] == selected[key]
    original_dir = baselines / f'B0_{meta["base_head"]}' / f'fold-{meta["fold"]}_seed-{meta["seed"]}'
    assert meta['base_checkpoint_sha256'] == sha256(original_dir / 'head.pt')
    original = pd.read_csv(original_dir / 'predictions.csv').set_index('trial_id')
    baseline = p.drop_duplicates('trial_id').set_index('trial_id').sort_index()
    np.testing.assert_array_equal(baseline.b0_prediction, original.loc[baseline.index].prediction)
    prior_dir = root / 'stage_b/runs' / f'fold-{meta["fold"]}_seed-{meta["seed"]}'
    assert meta['stage_b_head_results_sha256'] == sha256(prior_dir / 'subjects.csv')
    prior = pd.read_csv(prior_dir / 'subjects.csv').query('variant == "head"').set_index(['dataset', 'subject'])
    prior_predictions = pd.read_csv(prior_dir / 'predictions.csv').query('variant == "head"').set_index('trial_id')
    assert set(map(tuple, s[['dataset', 'subject']].drop_duplicates().to_numpy())) == set(map(tuple, folds[meta['fold']]['test']))
    assert set(s.variant) == set(cfg['variants'])
    assert not s.duplicated(['variant', 'dataset', 'subject']).any()
    assert not p.duplicated(['variant', 'trial_id']).any()
    for row in s.itertuples():
        g = p[(p.variant == row.variant) & (p.dataset == row.dataset) & (p.subject == row.subject)]
        pd.testing.assert_series_equal(g.set_index('trial_id').label.sort_index(), expected[(row.dataset, row.subject)])
        h = prior_predictions.loc[g.trial_id]
        np.testing.assert_array_equal(g.label, h.label)
        np.testing.assert_allclose([ba(g.label, g.b0_prediction), ba(g.label, g.prediction), ba(h.label, h.prediction)],
                                   [row.b0_ba, row.ba, row.head_ba], atol=1e-14, rtol=0)
        assert abs(row.head_ba - prior.loc[(row.dataset, row.subject), 'ba']) < 1e-14
        assert abs(row.ba - row.b0_ba - row.gain) < 1e-14
        assert abs(row.ba - row.head_ba - row.gain_vs_head) < 1e-14
        count += 1; predictions += len(g)
assert runs == 25 and count == 235 * 5 * 5
result = {'status': 'pass', 'runs': runs, 'subject_seed_variant_rows': count,
    'trial_prediction_rows': predictions, 'checks': [
        'query IDs and labels equal frozen second half for every subject/variant/seed',
        'outer test and validation subjects equal frozen split',
        'all hyperparameters independently reselected from validation-only search',
        'every B0 prediction equals Stage A original checkpoint prediction',
        'Stage B head predictions use exactly the same query trials and labels',
        'all three BA scores and both differences recomputed for every row',
        'configuration, halves, split, base checkpoint, and prior head result hashes verified',
        'all runs recorded full forward without activation cache and unchanged base weights']}
save_json(root / 'stage_b2/report/result_verification.json', result)
print(json.dumps(result))
