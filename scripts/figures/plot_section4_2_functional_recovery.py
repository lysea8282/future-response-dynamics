"""Publication-only extraction and drawing of frozen structured-GRU recovery medians."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STEM = 'paper2_fig2_structured_gru_functional_recovery'
CHECKPOINTS = [291405, 291406, 291407, 291410, 291411, 291412, 291413, 291415, 291416]
HASHES = {'original.json': 'D2BB2803757B3C5D72842752A7CCF88B04622C336B0664C449FF4B87269F27A7',
          'extension.json': '8052BF2B44DC0BE0855C875B3ED3A3F34D31D6576C2020F89C6C4B67B713937F'}
FIELDS = ['checkpoint', 'stratum', 'median_normalized_functional_recovery', 'complete_assay_pass']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def validate_records(original, extension):
    if [c['checkpoint'] for c in original['checkpoints']] != CHECKPOINTS[:3]:
        raise ValueError('ORIGINAL_CHECKPOINT_ORDER_MISMATCH')
    if [c['checkpoint'] for c in extension['checkpoints']] != CHECKPOINTS[3:]:
        raise ValueError('EXTENSION_CHECKPOINT_ORDER_MISMATCH')
    if extension['retained_baseline_ineligible_seed'] != 291414:
        raise ValueError('BASELINE_EXCLUSION_MISMATCH')
    rows, cells = [], {}
    for checkpoint in original['checkpoints'] + extension['checkpoints']:
        cp = checkpoint['checkpoint']
        if set(checkpoint['strata']) != {'S1', 'S2'}:
            raise ValueError('STRATUM_SET_MISMATCH')
        for st in ('S1', 'S2'):
            cell = checkpoint['strata'][st]
            if cell['unit_count'] != 32 or cell['unit_times'] != 320:
                raise ValueError('CONFIRMATION_POPULATION_MISMATCH')
            if not all(math.isfinite(cell[k]) for k in ('recovery', 'capture')):
                raise ValueError('NONFINITE_METRIC')
            if type(cell['supports']) is not bool:
                raise ValueError('INVALID_COMPLETE_ASSAY_STATUS')
            if not (cell['gates']['capture'] is True and cell['gates']['functional'] is True):
                raise ValueError('TRANSPORT_FUNCTION_CORE_PARITY_CONFLICT')
            if cell['supports'] != all(cell['gates'].values()):
                raise ValueError('COMPLETE_ASSAY_GATE_INCONSISTENCY')
            cells[cp, st] = cell
            rows.append(dict(checkpoint=cp, stratum=st,
                median_normalized_functional_recovery=cell['recovery'], complete_assay_pass=cell['supports']))
    values = [r['median_normalized_functional_recovery'] for r in rows]
    failed = {(cp, st) for (cp, st), cell in cells.items() if not cell['supports']}
    parity = {
        'nine_checkpoints_eighteen_cells': len(cells) == 18 and len({cp for cp, st in cells}) == 9,
        'recovery_range_0_744_to_1_034': (f'{min(values):.3f}', f'{max(values):.3f}') == ('0.744', '1.034'),
        'seventeen_above_0_85': sum(v > .85 for v in values) == 17,
        'all_eighteen_transport_function_pass': all(c['gates']['capture'] and c['gates']['functional'] for c in cells.values()),
        'sixteen_complete_assay_pass': sum(c['supports'] for c in cells.values()) == 16,
        'exact_two_failure_cells_and_values': failed == {(291405,'S1'), (291413,'S2')}
            and tuple(f"{cells[291405,'S1'][k]:.3f}" for k in ('recovery','capture')) == ('1.034','0.938')
            and tuple(f"{cells[291413,'S2'][k]:.3f}" for k in ('recovery','capture')) == ('0.965','0.977'),
        'baseline_ineligible_291414_excluded': all(cp != 291414 for cp, st in cells),
    }
    if not all(parity.values()):
        raise ValueError('SOURCE_PARITY_CONFLICT: ' + json.dumps(parity))
    result = dict(decision='PASS', checks=parity, eligible_checkpoint_order=CHECKPOINTS,
        cell_count=len(cells), min_recovery=min(values), max_recovery=max(values),
        above_0_85=sum(v > .85 for v in values), transport_function_pass=18, complete_assay_pass=16,
        failed_cells=[dict(checkpoint=cp, stratum=st, recovery=cells[cp,st]['recovery'],
            capture=cells[cp,st]['capture'], failed_gates=[k for k,v in cells[cp,st]['gates'].items() if not v])
            for cp,st in sorted(failed)])
    return rows, result

def load_rows(source_dir):
    sources = {}
    for name, expected in HASHES.items():
        p = source_dir / name
        if sha(p) != expected:
            raise ValueError('SOURCE_HASH_MISMATCH: ' + name)
        sources[name] = json.loads(p.read_text(encoding='utf-8'))
    return validate_records(sources['original.json'], sources['extension.json'])

def draw(rows, output_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':10,
        'xtick.labelsize':8.5, 'ytick.labelsize':9, 'legend.fontsize':8.5, 'axes.linewidth':.7,
        'pdf.fonttype':42, 'savefig.facecolor':'white'})
    fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=150)
    fig.subplots_adjust(left=.115, right=.98, bottom=.18, top=.80)
    styles = {'S1': ('#0072B2','o',-.13), 'S2': ('#D55E00','s',.13)}
    points = {}
    for row in rows:
        cp, st = row['checkpoint'], row['stratum']
        color, marker, offset = styles[st]
        x = CHECKPOINTS.index(cp) + offset
        y = row['median_normalized_functional_recovery']
        point, = ax.plot(x, y, linestyle='none', marker=marker, markersize=6.5,
            markeredgewidth=1.35, markeredgecolor=color,
            markerfacecolor=color if row['complete_assay_pass'] else 'white', zorder=4)
        assert float(point.get_ydata()[0]) == y
        points[cp,st] = (x,y)
    ax.axhline(1, color='#858585', linewidth=.85, linestyle=(0,(4,3)), zorder=1)
    annotations = []
    for key, xytext, color in [((291405,'S1'),(.65,1.067),'#0072B2'),
                               ((291413,'S2'),(5.85,.865),'#D55E00')]:
        annotations.append(ax.annotate(f'{key[0]}/{key[1]}', xy=points[key], xytext=xytext,
            fontsize=9, color=color, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', linewidth=.8, color=color, shrinkA=4, shrinkB=6)))
    ax.set_xlim(-.55, 8.55)
    ax.set_ylim(.70, 1.09)
    ax.set_xticks(range(9), [str(cp) for cp in CHECKPOINTS])
    ax.yaxis.set_major_locator(FixedLocator([.70,.80,.90,1.00]))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Baseline-eligible checkpoint', labelpad=9)
    ax.set_ylabel('Median normalized\nfunctional recovery', labelpad=9)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=3, width=.6)
    ax.grid(axis='y', color='#E5E5E5', linewidth=.5)
    ax.set_axisbelow(True)
    handles = [Line2D([],[],color=c,marker=m,linestyle='none',markersize=6,label=st)
               for st,(c,m,offset) in styles.items()]
    handles += [Line2D([],[],color='#444444',marker='o',linestyle='none',markersize=6,
        markerfacecolor=face,label=label) for face,label in [('#444444','Complete assay passes'),
        ('white','Core passes; complete assay fails')]]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.965), ncol=4,
        frameon=False, handletextpad=.5, columnspacing=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [legend, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(),
                   *ax.get_yticklabels(), *annotations]:
        b=artist.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1:
            raise ValueError('TEXT_OUTSIDE_CANVAS')
    if legend.get_window_extent(renderer).y0 <= ax.bbox.y1:
        raise ValueError('LEGEND_OVERLAPS_PLOT')
    output_root.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_root/(STEM+'.pdf'),metadata={'Title':'Structured-GRU functional recovery',
        'Author':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(output_root/(STEM+'.png'),dpi=600,metadata={'Software':'Paper2 frozen recovery figure'})
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=ROOT/'results/accepted_summaries')
    parser.add_argument('--output-root',type=Path,default=ROOT)
    parser.add_argument('--parity-output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        rows, parity = load_rows(args.sources)
        print('SOURCES: '+json.dumps({str(args.sources/name):value for name,value in HASHES.items()}))
        print('PARITY: '+json.dumps(parity))
        if args.parity_output:
            args.parity_output.parent.mkdir(parents=True,exist_ok=True)
            args.parity_output.write_text(json.dumps(parity,indent=2)+'\n',encoding='utf-8')
        if args.check_only:
            return 0
        data=args.output_root/'results/paper_figures'/(STEM+'.csv')
        data.parent.mkdir(parents=True,exist_ok=True)
        with data.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
            writer.writeheader();writer.writerows(rows)
        with data.open(encoding='utf-8',newline='') as stream:
            readback=list(csv.DictReader(stream))
        assert len(readback)==18
        for actual, expected in zip(readback,rows):
            assert all(actual[k]==str(expected[k]) for k in FIELDS)
        draw(rows,args.output_root/'figures')
        for p in [data, args.output_root/'figures'/(STEM+'.pdf'),args.output_root/'figures'/(STEM+'.png')]:
            print('OUTPUT: '+str(p)+' SHA256='+sha(p))
        import matplotlib
        print('ENVIRONMENT: '+json.dumps(dict(python=sys.version,executable=sys.executable,matplotlib=matplotlib.__version__)))
        return 0
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
        print(str(exc),file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())
