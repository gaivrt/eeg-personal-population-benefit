# Supplementary Data 1

All 904 original S5 comparison rows (T0001-T0904), unchanged in identifier order and recorded values. No training, inference, statistical tests or multiplicity correction were rerun. CSV is UTF-8 with a byte-order mark. The original 16 fields retain their exact strings, including blanks; five English explanation fields are appended. XLSX has the same rows, with numeric cells for saved numbers and separate Columns, Families and Notation sheets. The CSV preserves the exact source decimal strings independently of spreadsheet display precision.

Numerical results retain their original precision. Missing entries remain blank. All-trial rows, query-informed priors and overlapping cohort summaries retain their distinct scope. Original multi-component decisions remain in the supplementary PDF.

## Columns

| Column | Meaning | Unit | Interpretation |
|---|---|---|---|
| id | Original complete comparison identifier | identifier | Retained verbatim; stage and condition components are defined in Notation. |
| stage | Source analysis family | identifier | Case-sensitive; distinct from condition identifiers. |
| hypothesis | Original contrast description | text | Original language retained; see contrast_description for English subtraction direction. |
| condition | Original parameter, control and comparison fields | text | Semicolon-separated key=value fields; notation explains each key and value. |
| dataset | Dataset or pooled cohort | text | Overlapping pooled and subset rows are not independent replications. |
| n | Subjects contributing to the row | people | Seeds are averaged within subject before summaries; source cohort determines eligibility. |
| median_pp | Saved median paired contrast | percentage points | Subtraction follows hypothesis; blank means unavailable, not zero. |
| mean_pp | Saved mean paired contrast | percentage points | Subtraction follows hypothesis; blank means unavailable, not zero. |
| p_raw | Saved raw paired Wilcoxon p-value | unitless, 0 to 1 | Direction in test_direction; zero differences removed, normal approximation with tied ranks in the source implementations. |
| p_adjusted | Saved Holm-adjusted p-value | unitless, 0 to 1 | Only the original family in correction_family/correction_scope; blank is not an adjusted p-value of one or zero. |
| rule | Original row-level rule | text | Original language retained; no new decision threshold applied. |
| status | Original interpretation/decision record | text | Retained verbatim; a row-level difference does not by itself satisfy a multi-component original gate. |
| source | Original relative source-file identifier | text | Provenance, not a live external link; raw separators retained. |
| source_row | Original one-based source data-row locator | row index | Header excluded; blank where a row aggregates multiple source records. |
| aggregation | Original aggregation description | text | Blank where no extra aggregation is specified. |
| print_id | Stable supplementary row identifier | identifier | T0001 through T0904; retained without renumbering. |
| contrast_description | English contrast description | text | Positive values favor the left-hand condition. In adapt_gain rows the contrast compares calibration increments. |
| test_direction | Alternative hypothesis or descriptive status | text | two-sided: difference is nonzero; greater: left minus right is positive; descriptive: no test. |
| correction_family | Original adjustment-family identifier | identifier | H01-H12 are local documentation keys for the preserved families; none means no adjusted value in this row. |
| correction_scope | Members and scope of the original family | text | No correction is applied across the combined 904 rows. |
| information_access_note | Row-specific access or scoring caveat | text | Blank means no additional caveat beyond Methods and the source-family definitions. |

## Original correction families

| Family | Scope | Original source |
|---|---|---|
| H01 | Holm: 6 comparisons, 2 head types x 3 datasets. | reports/stage_a/baselines/wilcoxon.csv |
| H02 | Holm: all 25 rows, 5 variants x 5 overlapping cohort summaries. | reports/stage_b/report/summary.csv |
| H03 | Holm: 5 cohort summaries of selected FiLM versus tuned head. | reports/stage_b/report/film_vs_head.csv |
| H04 | Holm: 15 rows, 3 fixed FiLM/shift variants x 5 cohort summaries. | reports/stage_b/report/fixed_film_vs_head.csv |
| H05 | Holm: 12 rows, 2 alignment conditions x 2 heads x 3 datasets. | reports/stage_b/report/ea_tests.csv |
| H06 | Holm: all 25 full-forward variant versus frozen-head rows. | reports/stage_b2/report/summary.csv |
| H07 | Holm: all 25 full-forward variant versus tuned-head rows; separate from the four-variant directional decision family. | reports/stage_b2/report/vs_finetuned_head.csv |
| H08 | Holm: 4 pooled comparisons, 2 context families x 2 controls. | reports/stage_c/report/criteria_tests.csv |
| H09 | Holm: 60 comparisons, 2 context families x 6 controls x 5 cohort summaries. | reports/stage_c/report/diagnostic_tests.csv |
| H10 | Holm: 98 later-half condition/cohort comparisons; all-trial compatibility rows are excluded. | reports/stage_c/report/summary.csv |
| H11 | Holm: 15 two-sided own-versus-swapped comparisons, 3 parameter variants x 5 cohort summaries. | reports/stage_c2/report/summary.csv |
| H12 | Holm: 3 pooled rest-neighbor comparisons; dataset and subset rows are unadjusted. | reports/stage_c2/report/diagnostic2_tests.csv |
| none | No adjusted p-value supplied; not a combined adjustment family. | Row-specific source field |

The Notation sheet reproduces the S0 code key. Direction and correction membership are documented per row; raw p-values from later analyses are not retrospectively replaced by separately reported joint-gate values. The original Chinese hypothesis/rule/status text is preserved to avoid silently revising archived decisions.
