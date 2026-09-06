"""Display certified stored results after byte verification, without recalculating metrics."""
from pathlib import Path
import csv
import json
from verify_artifacts import inspect

def main():
    root=Path(__file__).resolve().parents[1]
    errors=inspect(root)
    if errors:raise SystemExit('\n'.join(errors))
    result=json.loads((root/'provenance/RESULT.json').read_text(encoding='utf-8'))
    print(result['overall_classification'])
    print('checkpoint supports:',json.dumps(result['checkpoint_support_counts'],sort_keys=True))
    for name in ('ALIGNMENT_CHECKPOINT_SUMMARY.csv','CAUSAL_CHECKPOINT_STRATUM_ROUTE_SUMMARY.csv','FULL_DOSE_SUMMARY.csv'):
        print('\n'+name)
        with (root/'results'/name).open(encoding='utf-8',newline='') as stream:
            for row in csv.DictReader(stream):print(json.dumps(row,sort_keys=True))

if __name__=='__main__':main()
