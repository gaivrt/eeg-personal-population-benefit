"""Readable Methods grid from recorded configurations; no fitting or inference."""
from common import *

def compact_grids():
    rows=[]
    def get(name,pointer):
        path='configs/'+name+'.yaml';source(path)
        value=yaml.safe_load((ROOT/path).read_text(encoding='utf-8'))
        for part in pointer.strip('/').split('/'):value=value[part]
        def register(v,p):
            key='config:'+path+p
            if isinstance(v,(float,int)) and not isinstance(v,bool) and key not in KEYS:
                add(key,v,'protocol',path,p,p,'recorded configuration scalar','Methods: Table 4')
            elif isinstance(v,list):
                for i,x in enumerate(v):register(x,p+'/'+str(i))
        register(value,pointer)
        if value is None:return 'none'
        if isinstance(value,list):return ', '.join(str(x) if not isinstance(x,list) else '-'.join(map(str,x)) for x in value)
        return str(value)
    def row(group,label,name,*pointers):
        rows.append([group,label,'; '.join(get(name,p) for p in pointers)])
    row('Partitions','Folds; seed values','stage_a','/folds','/seeds')
    row('Task head','Learning rates; epochs; batch','stage_a','/training/learning_rates','/training/epochs','/training/batch_size')
    row('Task head','Weight decay; MLP hidden width','stage_a','/training/weight_decay','/training/mlp_hidden')
    row('CBraMod input','Hz; high-pass Hz; rest segment s','stage_a','/preprocessing/sfreq','/preprocessing/highpass_hz','/preprocessing/rest_segment_seconds')
    row('Mains notch','PhysioNet; Dreyer; Cho (Hz)','stage_a','/datasets/PhysionetMI/notch_hz','/datasets/Dreyer2023/notch_hz','/datasets/Cho2017/notch_hz')
    row('Personal diagnostics','Earlier FiLM variants','stage_b','/variants')
    row('Personal diagnostics','Extended variants','stage_b2','/variants')
    row('Personal diagnostics','FiLM rate; head/LoRA rate','stage_b','/learning_rates','/head_learning_rates')
    row('Personal diagnostics','Steps; weight decay; batch','stage_b2','/steps','/weight_decays','/batch_size')
    row('Context','Minimum duration s; bands Hz','stage_c','/context/min_seconds','/context/bands_hz')
    row('Context','Band-filter order','stage_c','/context/filter_order')
    row('Context','Feature projection; hidden; latent','stage_c','/encoder/projection','/encoder/hidden','/encoder/z_dim')
    row('Context','Generator hidden; FiLM bound','stage_c','/generator_hidden','/film_bound')
    row('Context LoRA','Bases; rank; output-factor init SD','stage_c','/lora/bases','/lora/rank','/lora/b_init_std')
    row('Context training','Learning rates; candidate updates','stage_c','/training/learning_rates','/training/episodes')
    row('Context training','Margin weights; margins; zero ablation','stage_c','/training/lambdas','/training/margins','/training/ablation_lambda')
    row('Context training','Batch; context dropout; decay; clip','stage_c','/training/query_batch','/training/context_dropout','/training/weight_decay','/training/grad_clip_norm')
    row('Personal fitting','FiLM and mixture-offset rates; LoRA rates','stage_c2','/learning_rates/film_offset','/learning_rates/lora8')
    row('Personal fitting','Steps; weight decay; batch; rank','stage_c2','/steps','/weight_decays','/batch_size','/lora_rank')
    row('Exchange','Donor draws','stage_c2','/swap_draws')
    row('Few-shot fitting','Total label budgets','stage_c3','/few_shot/shots')
    row('Neighbors','Primary k; secondary k; random draws','stage_c3','/diagnostic3/primary_k','/diagnostic3/secondary_k','/diagnostic3/random_draws')
    row('Prior','Neighbors k; shrinkage alpha','stage_c4','/ks','/alphas')
    row('Prior','LoRA labels; penalty mu','stage_c4','/variants/lora8/shots','/variants/lora8/mu')
    row('Prior','FiLM labels; penalty mu','stage_c4','/variants/film_offset/shots','/variants/film_offset/mu')
    row('Prior','Steps; weight decay; batch','stage_c4','/steps','/weight_decay','/batch_size')
    row('Prior','Test draws; validation draws','stage_c4','/random_draws','/validation_random_draws')
    row('G continuation','Additional-step multiplier; check period','stage_w','/population/additional_steps_multiplier','/population/check_every_steps')
    row('G continuation','Patience checks; minimum CE change','stage_w','/population/early_stop_patience_checks','/population/early_stop_min_delta')
    row('G plateau','Window checks; relative loss range','stage_w','/population/plateau_window_checks','/population/plateau_relative_range')
    row('Meta inner loop','SGD steps; learning rates','stage_m','/inner/ks','/inner/learning_rates')
    row('Meta inner pilot','Scanned learning rates; labels','stage_m','/inner/pilot/scan','/inner/pilot/shots')
    row('Meta outer loop','Episode labels; candidate updates','stage_m','/outer/episode_shots','/outer/steps')
    row('Meta outer loop','Learning rate; decay; gradient clip','stage_m','/outer/learning_rate','/outer/weight_decay','/outer/grad_clip_norm')
    row('Meta outer loop','Subjects per batch; M2 log-rate LR','stage_m','/outer/meta_batch_subjects','/outer/m2_log_lr_learning_rate')
    row('REVE encoder','Depth; width; patch samples; stride','cross_model_x','/models/REVE/depth','/models/REVE/width','/models/REVE/patch_samples','/models/REVE/patch_stride')
    row('LaBraM encoder','Depth; width; patch samples; stride','cross_model_x','/models/LaBraM/depth','/models/LaBraM/width','/models/LaBraM/patch_samples','/models/LaBraM/patch_stride')
    row('LaBraM input','High-pass; low-pass; notch (Hz)','cross_model_x','/preprocessing/LaBraM/highpass_hz','/preprocessing/LaBraM/lowpass_hz','/preprocessing/LaBraM/notch_hz_all_datasets')
    row('REVE input','High-pass Hz; sampling Hz','cross_model_x','/preprocessing/REVE/starter_and_Lee_highpass_hz','/preprocessing/sfreq')
    row('New-model LoRA','Rank; scale/rank; dropout','cross_model_x','/adaptation/rank','/adaptation/alpha_over_rank','/adaptation/dropout')
    row('New-model population','Learning rates; candidate updates','cross_model_x','/population/G/learning_rates','/population/G/initial_checkpoint_candidates')
    row('New-model population','Batch; decay; gradient clip','cross_model_x','/population/G/batch_size','/population/G/weight_decay','/population/G/gradient_clip')
    row('New-model plateau','Checks; relative CE range; floor','cross_model_x','/convergence/window_checks','/convergence/loss_relative_range','/convergence/loss_absolute_range_floor')
    row('New-model plateau','Accuracy/BA range; best-gain threshold','cross_model_x','/convergence/accuracy_and_BA_absolute_range','/convergence/accuracy_and_BA_min_delta')
    row('Budget extension','Update multipliers; check period','population_budget','/multipliers','/convergence/check_every_added_steps')
    row('Budget extension','Directional-test alpha','population_budget','/statistics/alpha')
    row('BNCI input','Window endpoints after trial start s','stage_w_followup','/bnci/trial_seconds_after_trial_start')
    path='reports/stage_a/baselines/report.md'
    source(path)
    add('journal.EA.shrinkage',0.001,'protocol',path,'line 30','fixed diagonal shrinkage','value transcribed from recorded baseline report','Methods: Table 4')
    rows.append(['Euclidean alignment','Fixed diagonal shrinkage','0.001'])
    table_file('hyperparameters',['Component','Setting (semicolon order)','Value / grid'],rows,widths='p{1.15in}p{2.7in}p{2.75in}',long=True)
