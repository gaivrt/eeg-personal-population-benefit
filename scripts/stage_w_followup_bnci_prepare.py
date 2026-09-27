"""Prepare approved BNCI session E after download and original-GDF rest verification."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from subject_context.data import make_splits
from subject_context.stage_a_common import ROOT,require_compute,save_json,scratch_root,sha256
from subject_context.stage_a_preprocess import filtered_array,save_signal,segment_rest
from subject_context.stage_b import temporal_halves

CHANNELS='Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz'.split()

def main():
    require_compute()
    root=scratch_root();configuration=ROOT/'configs/stage_w_followup.yaml'
    cfg=yaml.safe_load(configuration.read_text())['bnci'];dataset=cfg['dataset']
    prep=root/'stage_w_followup/bnci'
    assert not (prep/'prepared.json').exists(),'Inspect existing prepared data instead of overwriting'
    manifest=json.loads((prep/'download_manifest.json').read_text())
    gdf=json.loads((prep/'rest_gdf_audit.json').read_text())
    assert manifest['status']==gdf['status']=='complete' and len(manifest['files'])==len(gdf['subjects'])==9
    rests={r['subject']:r for r in gdf['subjects']}
    common=json.loads((ROOT/'configs/stage_a_channels.json').read_text())['readout_common']
    readout=[c for c in common if c in CHANNELS];assert len(readout)==19
    audits=[]
    for entry in manifest['files']:
        subject=entry['subject'];path=Path(entry['path']);assert sha256(path)==entry['sha256']
        assert rests[subject]['rest_code']==276 and rests[subject]['all_25_channels_equal']
        blocks=np.atleast_1d(loadmat(path,simplify_cells=True)['data'])
        assert len(blocks)==9
        assert all(np.asarray(b['y']).size==0 for b in blocks[:3])
        out=root/'processed'/dataset/f'sub-{subject:03d}'
        out.mkdir(parents=True,exist_ok=True);assert not (out/'metadata.json').exists()
        tasks,rows=[],[]
        for run,block in enumerate(blocks[3:],start=1):
            assert block['fs']==250 and list(block['classes'])==['left hand','right hand','feet','tongue']
            raw=np.asarray(block['X'],dtype=np.float64);assert raw.shape[1]==25 and np.isfinite(raw).all()
            trials=np.asarray(block['trial'],dtype=int).reshape(-1)
            labels=np.asarray(block['y'],dtype=int).reshape(-1)
            artifacts=np.asarray(block['artifacts'],dtype=int).reshape(-1)
            assert len(trials)==len(labels)==len(artifacts)==48 and np.all(np.diff(trials)>0)
            assert np.bincount(labels,minlength=5).tolist()==[0,12,12,12,12]
            signal=filtered_array(raw[:,:22].T*1e-6,250,CHANNELS,dataset,cfg)
            for index,(sample,label) in enumerate(zip(trials,labels)):
                if label not in [1,2]:continue
                onset=(int(sample)-1)/250+2 # MATLAB trial onset is one-based; MI starts at cue, +2 seconds.
                begin=round(onset*200);assert 0<=begin and begin+800<=signal.shape[1]
                tasks.append(signal[:,begin:begin+800])
                rows.append({'trial_id':f'{dataset}:{subject:03d}:{len(rows):04d}','label':int(label==2),
                             'source_file':str(path),'source_block_zero':run+2,'run':run,
                             'source_event_index':index,'source_event_sample_one_based':int(sample),
                             'source_trial_label':int(label),'expert_artifact_flag':int(artifacts[index]),
                             'start_seconds':onset,'stop_seconds':onset+4,'protocol_order':run,'quality_valid':True})
        table=pd.DataFrame(rows);assert len(table)==144
        values=np.stack(tasks);assert values.shape==(144,22,800) and np.isfinite(values).all()
        save_signal(out/'tasks.npy',values);np.save(out/'labels.npy',table.label.to_numpy(dtype=np.int64))
        np.save(out/'task_valid.npy',np.ones(144,dtype=bool));table.to_csv(out/'trials.csv',index=False)
        support,query,counts=temporal_halves(table)
        assert support.tolist()==list(range(72)) and query.tolist()==list(range(72,144))
        assert [counts[k] for k in ['fit_left','fit_right','query_left','query_right']]==[36,36,36,36]
        rest=np.asarray(blocks[0]['X'],dtype=np.float64)[:,:22]
        assert len(rest)/250==rests[subject]['rest_seconds'] and np.isfinite(rest).all()
        windows=segment_rest(filtered_array(rest.T*1e-6,250,CHANNELS,dataset,cfg))
        assert len(windows)*4>=30
        save_signal(out/'rest_open.npy',windows);np.save(out/'rest_open_valid.npy',np.ones(len(windows),dtype=bool))
        rest_meta={'open':{'source_file':str(path),'source_block_zero':0,'eye_state':'open','duration_seconds':len(rest)/250,
                           'usable_seconds':len(windows)*4,'protocol_order':0,'verified_original_gdf_code':276}}
        record={'dataset':dataset,'subject':subject,'session':2,'status':'complete','counts':counts,
                'raw_sha256':entry['sha256'],'w_config_sha256':sha256(configuration),
                'trial_table_sha256':sha256(out/'trials.csv'),'fit_indices':support.tolist(),'query_indices':query.tolist(),
                'rest_available':True,'readout_channels':readout,'expert_flagged_trials_retained':int(table.expert_artifact_flag.sum())}
        save_json(out/'metadata.json',{'dataset':dataset,'subject':subject,'session':2,'channels':CHANNELS,'sfreq':200,
                    'units':'uV/100','dtype':'float16','rest':rest_meta,'source_files':[{'path':str(path),'sha256':entry['sha256']}],
                    'checks':record})
        save_json(out/'audit_w_followup.json',record);audits.append(record)
        print(json.dumps({'subject':subject,'counts':counts,'rest_seconds':len(rest)/250,'expert_flagged':record['expert_flagged_trials_retained']}),flush=True)
    splits=make_splits([(dataset,s) for s in range(1,10)],5,.10,20260925)
    save_json(prep/'splits.json',{'folds':[dict(zip(['train','validation','test'],s)) for s in splits]})
    save_json(prep/'halves.json',{'subjects':audits})
    save_json(prep/'preprocessing.json',{'subjects':audits,'errors':[]})
    save_json(prep/'prepared.json',{'status':'complete','dataset':dataset,'session':2,'subjects':9,'rest_available_subjects':9,
                    'readout_channels':readout,'w_config_sha256':sha256(configuration),'available_bytes_after':shutil.disk_usage(root).free})

if __name__=='__main__':main()
