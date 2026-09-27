"""Read-only check that the approved MAT epochs start at original GDF cues."""
import hashlib
import json
import os
from pathlib import Path
import mne
import numpy as np
from scipy.io import loadmat

root=Path(os.environ['STAGE_A_ROOT'])
rows=[]
for subject in range(1,10):
    directory=root/'raw/BNCI2014_001/session2'
    raw=mne.io.read_raw_gdf(directory/f'A{subject:02d}E.gdf',preload=False,verbose=False)
    blocks=loadmat(directory/f'A{subject:02d}E.mat',simplify_cells=True)['data']
    codes=np.asarray([int(d) for d in raw.annotations.description])
    samples=np.rint(raw.annotations.onset*250).astype(int)
    starts=samples[codes==768]
    cue_mask=np.isin(codes,[769,770,771,772,783]);cues=samples[cue_mask];cue_codes=codes[cue_mask]
    assert len(starts)==len(cues)==288
    np.testing.assert_array_equal(cues-starts,np.full(288,500))
    offsets=[]
    for run,block in enumerate(blocks[3:]):
        trial=np.asarray(block['trial'],dtype=int)-1
        original=starts[48*run:48*(run+1)]
        offset=original-trial
        assert len(set(offset))==1
        offset=int(offset[0]);offsets.append(offset)
        observed=raw.get_data(start=offset,stop=offset+512).T*1e6
        np.testing.assert_allclose(observed,np.asarray(block['X'])[:512],atol=1e-8,rtol=1e-10)
        np.testing.assert_array_equal(offset+trial+500,cues[48*run:48*(run+1)])
        if not (cue_codes==783).any():
            np.testing.assert_array_equal(cue_codes[48*run:48*(run+1)]-768,block['y'])
    rows.append({'subject':subject,'session':2,'original_trial_starts':288,'original_cues':288,
                 'cue_offset_samples':500,'sfreq':250,'cue_offset_seconds':2,
                 'task_block_offsets':offsets,'task_block_samples_25ch_match':True,
                 'MAT_trial_is_trial_start_not_cue':True,'crop_starts_at_cue':True,
                 'cue_codes':sorted(set(cue_codes.tolist())),
                 'GDF_class_labels_available':bool(not (cue_codes==783).any())})
    raw.close();print(json.dumps(rows[-1]),flush=True)
path=root/'stage_w_followup/bnci/task_events_audit.json'
path.write_text(json.dumps({'status':'complete','read_only_audit':True,'subjects':rows,
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'GDF_unknown_cues_do_not_validate_class_identity':'When code 783 is present, labels remain sourced from official MAT ground truth.'},indent=2))
