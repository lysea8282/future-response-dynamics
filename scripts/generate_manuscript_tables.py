"""Format current Tables 1-4 from released frozen results; no model execution."""
from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

def rows(root, name):
    with (root / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def build(root=ROOT):
    lstm = json.loads((root / 'results/accepted_summaries/lstm.json').read_text(encoding='utf-8-sig'))
    t1 = []
    for checkpoint in lstm['checkpoints']:
        row = {'Checkpoint': checkpoint['checkpoint']}
        for st in ('S1', 'S2'):
            for metric in ('capture', 'recovery'):
                row[f'{st} {metric}'] = f"{checkpoint['strata'][st][metric]:.3f}"
        t1.append(row)
    alignment = rows(root, 'results/ALIGNMENT_CHECKPOINT_SUMMARY.csv')
    separation = rows(root, 'results/DELTA_TRIAD_SUMMARY.csv')
    causal = rows(root, 'results/CAUSAL_CHECKPOINT_STRATUM_ROUTE_SUMMARY.csv')
    dose = rows(root, 'results/FULL_DOSE_SUMMARY.csv')
    a = {(int(r['checkpoint']), r['stratum']): r for r in alignment}
    d = {(int(r['checkpoint']), r['stratum']): r for r in separation if r['metric'] == 'Delta_PF_Align'}
    c = {(int(r['checkpoint']), r['stratum'], r['route']): r for r in causal}
    t2, t3 = [], []
    for seed in sorted({key[0] for key in a}):
        r2, r3 = {'Checkpoint': seed}, {'Checkpoint': seed}
        for st in ('S1', 'S2'):
            r2[f'{st} CF-directed gain'] = f"{float(a[seed, st]['median_Delta_align']):.4f}"
        for st in ('S1', 'S2'):
            r2[f'{st} P-F separation'] = f"{float(d[seed, st]['median']):.4f}"
            for route in ('P', 'CF'):
                r3[f'{st} {route}'] = f"{float(c[seed, st, route]['median_spearman']):.6f}"
        t2.append(r2); t3.append(r3)
    thresholds = json.loads((root / 'configs/SCIENTIFIC_CONTRACT.json').read_text())['full_dose_thresholds']
    t4 = []
    for r in dose:
        valid = (r['numerical_valid'] == 'True'
                 and float(r['median_E_int32']) < thresholds['median_E_int32_lt']
                 and float(r['median_I32_I16_relative']) < thresholds['median_I32_I16_relative_lt'])
        passed = valid and float(r['median_E_local']) > thresholds['median_E_local_gt'] and float(r['median_C_R_midpoint']) > thresholds['median_C_R_midpoint_gt']
        t4.append({'Checkpoint': int(r['checkpoint']), 'Stratum': r['stratum'],
                   'Full-dose residual': f"{float(r['median_E_local']):.4f}",
                   'Midpoint chord deviation': f"{float(r['median_C_R_midpoint']):.4f}",
                   'Numerical valid': 'Yes' if valid else 'No', 'Cell': 'Pass' if passed else 'Fail'})
    numerical = {key: {'min': min(float(r[key]) for r in dose), 'max': max(float(r[key]) for r in dose)}
                 for key in ('median_E_int32', 'median_I32_I16_relative')}
    return {'table1_lstm': t1, 'table2_alignment': t2, 'table3_perturbation': t3, 'table4_finite_dose': t4}, numerical

def generate(root, output):
    tables, numerical = build(root)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in tables.items():
        keys = list(data[0])
        with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n'); writer.writeheader(); writer.writerows(data)
        markdown = '| ' + ' | '.join(keys) + ' |\n| ' + ' | '.join('---' for _ in keys) + ' |\n'
        markdown += ''.join('| ' + ' | '.join(str(r[k]) for k in keys) + ' |\n' for r in data)
        (output / (name + '.md')).write_text(markdown, encoding='utf-8')
    (output / 'path_accounting.json').write_text(json.dumps(numerical, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'tables': list(tables), 'rows': [len(r) for r in tables.values()], 'path_accounting': numerical}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args(); generate(args.root.resolve(), args.output.resolve())
