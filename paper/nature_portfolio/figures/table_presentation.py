"""Captions and readable labels for supplementary tables; no numerical changes."""
import re

CAPTIONS = {
    'bootstrap_intervals': ('bootstrap_intervals', 'Subject-bootstrap intervals for all main-text comparisons. Population balanced accuracy is in percent; paired differences are in percentage points.'),
    'fewshot_distributions': ('fewshot_distributions', 'CBraMod few-label calibration: paired subject means and between-subject standard deviations. Gains are in percentage points.'),
    'related_work_full': ('related', 'Related evaluation questions, target settings and controls. The comparison concerns study designs and inspected controls, not an accuracy ranking.'),
    'gates': ('gates', 'Original joint decision rules, observed components and decisions. A favorable component does not override a failed joint rule.'),
    'convergence': ('convergence', 'CBraMod continuation records by dataset, parameter family, fold and seed. Update counts are additional continuation updates; no run confirms the joint plateau.'),
    'cross_model_effects': ('cross_model_effects', 'Mean and median core contrasts and pretraining-source-list status. Differences are in percentage points; seeds are averaged within subject.'),
    'cross_original_gates': ('cross_original_gates', 'Original pooled specificity decisions for REVE and LaBraM. The saved three-slot correction is retained.'),
    'cross_original_comparisons': ('cross_original_comparisons', 'REVE and LaBraM core contrasts by dataset. Means, medians and standard deviations are in percentage points; p-values are the saved descriptive raw values.'),
    'cross_model_stability': ('cross_model_stability', 'Descriptive consistency of few-label calibration across the prescribed label budgets.'),
    'cross_few_REVE': ('cross_few_reve', 'REVE few-label results. Gains are in percentage points; decline is the subject fraction losing more than two points.'),
    'cross_few_LaBraM': ('cross_few_labram', 'LaBraM few-label results. Gains are in percentage points; decline is the subject fraction losing more than two points.'),
    'cross_convergence': ('cross_convergence', 'REVE and LaBraM continuation records. Selected and executed updates are distinguished; the stopping rule does not establish a joint plateau.'),
    'budget_CBraMod': ('budget_cbramod', 'Completed CBraMod population-budget results. Population balanced accuracy is in percent; personal benefit and specificity are in percentage points.'),
    'budget_REVE': ('budget_reve', 'Completed REVE population-budget results. Population balanced accuracy is in percent; personal benefit and specificity are in percentage points.'),
    'budget_LaBraM': ('budget_labram', 'Completed LaBraM population-budget results. Population balanced accuracy is in percent; personal benefit and specificity are in percentage points.'),
    'budget_decisions': ('budget_decisions', 'Completed population-budget decision records with the original six-slot positive-gain correction.'),
    'budget_all_tests': ('budget_all_tests', 'All completed population-budget comparisons. Difference columns are in percentage points; decline columns are subject fractions.'),
    'budget_validation': ('budget_validation', 'Validation performance at each completed population-budget endpoint. Run-level means and standard deviations use equal fold/seed weights.'),
    'budget_CB_missing': ('budget_cb_missing', 'Subject and seed coverage in the initial CBraMod budget batch before the documented rerun.'),
    'budget_CB_runs': ('budget_cb_runs', 'Initial CBraMod budget results by fold and seed. Missing personal diagnostics remain unavailable.'),
    'budget_CB_failed_validation': ('budget_cb_failed_validation', 'Available validation observations from the failed initial CBraMod run. Later endpoints are not filled.'),
    'budget_initial_decisions': ('budget_initial_decisions', 'Initial incomplete-batch budget decisions and their original six-slot correction, separate from the completed analysis.'),
    'meta_divergence_inventory': ('meta_divergence_inventory', 'Recorded inner-loop numerical events by fold, seed and support-label count. Counts are events, not affected subjects.'),
}

def present(name, text):
    if name in CAPTIONS and '\\begin{tabular}' in text:
        text=text.replace('\\begin{tabular}','\\begin{longtable}').replace('\\end{tabular}','\\end{longtable}')
    if name in CAPTIONS and '\\caption{' not in text:
        label, caption = CAPTIONS[name]
        end = text.index('\n')
        text = text[:end+1] + '\\caption{' + caption + '}\\label{tab:' + label + '}\\\\\n' + text[end+1:]
    # Replace whole cell labels only, preserving result values and source identifiers.
    replacements = {
        r'm\_\allowbreak{}film': 'FiLM population',
        r'm\_\allowbreak{}lora': 'LoRA population',
        r'CE\_\allowbreak{}patience': 'Loss patience',
        r'step\_\allowbreak{}budget': 'Update cap',
        'starter': 'Core cohort',
    }
    for a,b in replacements.items():
        text=text.replace(a,b)
    if name == 'cross_convergence':
        text=text.replace('\\begin{longtable}{llllllll}', '\\begin{longtable}{llrrrrp{1.05in}l}')
        text=text.replace('G selected', 'Selected updates')
    if name == 'bootstrap_intervals':
        for a,b in [('nearest\\_\\allowbreak{}gain','Nearest - population'),('random\\_\\allowbreak{}gain','Random - population'),(' difference &',' Nearest - random &')]:
            text=text.replace(a,b)
    return text
