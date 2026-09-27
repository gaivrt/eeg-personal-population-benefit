"""Read-only reporting helpers. No model loading, fitting, inference, or tests."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent.parent
OUT = PAPER / 'figures' / 'generated'
TABLES = PAPER / 'tables'
for directory in (OUT, TABLES):
    directory.mkdir(parents=True, exist_ok=True)
NUMBERS, SOURCES, FACTS, CHECKS = [], {}, {}, []
KEYS = set()
INK = '#25364A'
BLUE = '#3F6489'
TEAL = '#337F7A'
COPPER = '#B47850'
GRAY = '#9AA6B2'
COLORS = {'B3': GRAY, 'G': BLUE, 'R2': COPPER, 'own': TEAL, 'swap': COPPER}
MODEL_COLORS = [BLUE, TEAL, COPPER]
LAYOUT_CHECKS = []
DATASETS = ['PhysionetMI', 'Dreyer2023', 'Cho2017', 'Lee2019_MI', 'BNCI2014_001']
NAMES = dict(zip(DATASETS, ['PhysioNet', 'Dreyer', 'Cho', 'Lee', 'BNCI']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9,
                     'axes.labelsize': 8.5, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 240, 'axes.axisbelow': True,
                     'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK,
                     'axes.edgecolor': '#778390', 'axes.linewidth': .65,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'xtick.major.size': 3, 'ytick.major.size': 3,
                     'xtick.major.width': .65, 'ytick.major.width': .65,
                     'axes.titlepad': 10, 'axes.labelpad': 5,
                     'grid.color': '#DCE3E9', 'grid.linewidth': .6,
                     'lines.linewidth': 1.45, 'lines.markersize': 4,
                     'legend.handlelength': 1.8, 'legend.columnspacing': 1.6,
                     'legend.frameon': False, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

def source(path):
    path = str(path).replace('\\', '/')
    if path not in SOURCES:
        SOURCES[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    return path

def read(path):
    path = source(path)
    df = pd.read_csv(ROOT / path)
    df['_record'] = np.arange(1, len(df) + 1)
    df.attrs['source'] = path
    return df

def read_json(path):
    source(path)
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))

def add(key, value, unit, paths, records, columns, computation, location, claim='', display=None):
    value = float(value)
    assert math.isfinite(value), (key, value)
    paths = [paths] if isinstance(paths, str) else paths
    paths = [p.replace('\\', '/') for p in paths]
    for path in paths: source(path)
    identifier = 'N' + hashlib.sha256(key.encode()).hexdigest()[:12]
    assert key not in KEYS, key
    KEYS.add(key)
    shown = display if display is not None else (str(int(value)) if unit == 'count' else f'{value:.3g}' if unit == 'p' else f'{value:.2f}')
    NUMBERS.append(dict(number_id=identifier, key=key, location=location, claim_id=claim,
                        value=repr(value), display_value=shown, unit=unit,
                        source_file=';'.join(paths), source_record=str(records), source_column=columns,
                        computation=computation, source_sha256=';'.join(SOURCES[p] for p in paths)))
    return value

def stat(df, column, key, location, claim='', op='mean', scale=100, unit='pp'):
    good = df[column].notna()
    values = df.loc[good, column].astype(float)
    operations = {'mean': values.mean, 'median': values.median, 'sd': lambda: values.std(ddof=1),
                  'q25': lambda: values.quantile(.25), 'q75': lambda: values.quantile(.75),
                  'min': values.min, 'max': values.max, 'count': lambda: len(values)}
    return add(key, operations[op]() * scale, unit, df.attrs['source'],
               ','.join(df.loc[good, '_record'].astype(str)), column,
               f'{scale} * {op}(nonmissing source rows); subject-level seed averages; paper/nature_portfolio/figures/common.py', location, claim)

def direct(df, index, column, key, location, claim='', unit='pp', scale=1):
    row = df.loc[index]
    return add(key, float(row[column])*scale, unit, df.attrs['source'], int(row['_record']), column,
               f'{scale} * source cell', location, claim)

def fact(name, key):
    row = next(r for r in NUMBERS if r['key'] == key)
    FACTS[name] = row

def check_unique(df, keys):
    assert not df.duplicated(keys).any(), (df.attrs.get('source'), keys)
    CHECKS.append({'check': 'unique_keys', 'source': df.attrs.get('source'), 'keys': keys, 'rows': len(df)})

def save(fig, name):
    # Fixed canvases expose overflow instead of silently enlarging/cropping the page.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        if ax.axison:
            texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label])
            for axis in [ax.xaxis,ax.yaxis]:
                lo,hi=sorted(axis.get_view_interval())
                texts.extend(t for pos,t in zip(axis.get_ticklocs(),axis.get_ticklabels()) if lo<=pos<=hi)
    failures=[]
    for t in texts:
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0 < 1 or b.y0 < 1 or b.x1 > fig.bbox.width-1 or b.y1 > fig.bbox.height-1:
            failures.append(t.get_text())
    for legend in list(fig.legends)+[ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        b=legend.get_window_extent(renderer)
        if b.x0<1 or b.y0<1 or b.x1>fig.bbox.width-1 or b.y1>fig.bbox.height-1:
            failures.append('legend outside canvas')
    clipped=[]
    for i,ax in enumerate(fig.axes):
        if not ax.axison:continue
        for dim,limits,observed in [('x',ax.get_xlim(),ax.dataLim.intervalx),('y',ax.get_ylim(),ax.dataLim.intervaly)]:
            if np.isfinite(observed).all() and (observed[0]<min(limits)-1e-8 or observed[1]>max(limits)+1e-8):
                clipped.append({'panel':i,'axis':dim,'data':observed.tolist(),'limits':list(limits)})
    assert not failures,(name,'text outside fixed figure canvas',failures)
    assert not clipped,(name,'data outside axes',clipped)
    LAYOUT_CHECKS.append({'figure':name,'canvas_inches':fig.get_size_inches().tolist(),
                          'visible_texts_checked':len(texts),'text_outside_canvas':failures,'data_outside_axes':clipped})
    fig.savefig(OUT / f'{name}.pdf', metadata={'Author': '', 'Creator': 'Matplotlib', 'CreationDate': None})
    fig.savefig(OUT / f'{name}.png')
    (PAPER/'figure_layout_checks.json').write_text(json.dumps(LAYOUT_CHECKS,indent=2),encoding='utf-8')
    plt.close(fig)

def tex_escape(s):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    return ''.join(replacements.get(c,c) for c in str(s)).replace(r'\_',r'\_\allowbreak{}').replace('−','-').replace('≥',r'$\geq$').replace('≤',r'$\leq$').replace('±',r'$\pm$')

def table_file(name, headers, rows, widths=None, long=False):
    with (TABLES / f'{name}.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.writer(f); writer.writerow(headers); writer.writerows(rows)
    env='longtable' if long else 'tabular'
    spec=widths or ('l'*len(headers))
    lines=[r'\begin{'+env+'}{'+spec+'}', r'\toprule', ' & '.join(headers)+r' \\', r'\midrule']
    if long: lines += [r'\endfirsthead',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead']
    lines += [' & '.join(tex_escape(v) for v in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{'+env+'}']
    from table_presentation import present
    (TABLES/f'{name}.tex').write_text(present(name,'\n'.join(lines)+'\n'),encoding='utf-8')

def finish():
    for path, expected in SOURCES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, f'Input changed during build: {path}'
    for name, row in FACTS.items():
        locations=[]
        for path in sorted((PAPER/'sections').glob('*.tex')):
            for line,text in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
                if r'\N{'+name+'}' in text:locations.append(str(path.relative_to(ROOT)).replace('\\','/')+':'+str(line))
        if locations:row['location']+=';'+';'.join(locations)
    fields=list(NUMBERS[0])
    with (PAPER/'numbers.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(NUMBERS)
    (PAPER/'facts.json').write_text(json.dumps(FACTS,indent=2),encoding='utf-8')
    # Macro values used in prose come from the same full-precision ledger as plots.
    def latex_value(row):
        s=row['display_value']
        if 'e' in s.lower():
            mantissa,exponent=s.lower().split('e')
            return r'\ensuremath{'+mantissa+r'\times10^{'+str(int(exponent))+'}}'
        return s
    (PAPER/'facts.tex').write_text('% Generated from preserved results. Do not edit.\n'+''.join(
        r'\expandafter\def\csname fact'+name+r'\endcsname{'+latex_value(r)+'}\n' for name,r in FACTS.items()),encoding='utf-8')
    manifest={'sources':SOURCES,'number_records':len(NUMBERS),'checks':CHECKS,
              'scope':'Existing results only; subject bootstrap of saved seed-averaged scores; no fitting, inference, or new hypothesis tests.',
              'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'PyYAML':yaml.__version__}}
    (PAPER/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (PAPER/'requirements-reporting.txt').write_text('\n'.join(f'{k}=={v}' for k,v in manifest['versions'].items() if k!='python')+'\npypdf==6.10.0\n',encoding='utf-8')
