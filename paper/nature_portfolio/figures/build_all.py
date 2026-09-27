"""One command rebuild: existing reports -> figures, tables, numeric ledger.

Run from any working directory: python paper/figures/build_all.py
This entry point never imports the research package or training scripts.
"""
import common
import main_figures
import tables
import cross_model_results
import budget_results
import revision_statistics

if __name__ == '__main__':
    main_figures.build()
    tables.build()
    cross_model_results.build()
    budget_results.build()
    revision_statistics.build()
    common.finish()
    print(f'Built figures/tables and {len(common.NUMBERS)} provenance records from {len(common.SOURCES)} source files.')
