"""Publication-only extraction from accepted audit freezes; no model/metric execution.

Usage: python -B generate_paper_tables.py --sources DIR --output NEW_DIR
Sources: original.json, extension.json, lstm.json, lstm_adequacy.csv.
The source package binds each file to its canonical audit and SHA256.
"""
from pathlib import Path
import argparse,csv,io,json,hashlib
TRAINED=[291405,291406,291407,291410,291411,291412,291413,291414,291415,291416]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,value):
    with p.open('x',encoding='utf-8',newline='') as f:f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csv_text(rows):
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return buf.getvalue()
def md(rows,title):
    keys=list(rows[0]);fmt=lambda x:format(x,'.6f') if isinstance(x,float) else str(x).replace('|','/').replace('\n',' ')
    return '# '+title+'\n\n| '+' | '.join(keys)+' |\n| '+' | '.join('---' for k in keys)+' |\n'+''.join('| '+' | '.join(fmt(r[k]) for k in keys)+' |\n' for r in rows)+'\nDisplayed floats rounded to six decimals; paired CSV retains accepted full precision. Evidence IDs resolve through the claim/evidence registry.\n'
def generate(sources,output):
    original=read(sources/'original.json');extension=read(sources/'extension.json');lstm=read(sources/'lstm.json')
    original_rows=original['checkpoints'];ext_rows=extension['checkpoints'];allcp={x['checkpoint']:x for x in original_rows+ext_rows}
    assert [x['checkpoint'] for x in original_rows]==TRAINED[:3]
    assert [x['checkpoint'] for x in ext_rows]==[291410,291411,291412,291413,291415,291416]
    assert extension['retained_baseline_ineligible_seed']==291414
    adequacy=list(csv.DictReader((sources/'lstm_adequacy.csv').open(encoding='utf-8-sig')))
    assert {int(r['seed']) for r in adequacy}=={392001,392002,392003}
    baseline={seed:all(r['baseline_eligible']=='True' and r['pass_']=='True' for r in adequacy if int(r['seed'])==seed) for seed in [392001,392002,392003]}
    rows=[];cells=[];bands=[]
    for cp in TRAINED:
        eid='E05' if cp in TRAINED[:3] else 'E06'
        if cp==291414:
            rows.append(dict(checkpoint=cp,baseline_eligibility='INELIGIBLE',S1_full_mechanism_support='NOT_EVALUATED',S2_full_mechanism_support='NOT_EVALUATED',checkpoint_support='NOT_EVALUATED',end_to_end_support=False,failure_reason='Frozen baseline health gate; mechanism not evaluated',canonical_source_audit='E08; E06 accepted audit'))
            continue
        d=allcp[cp];s=d['strata'];reason=[]
        for st,m in s.items():
            if not m['supports']:
                reason.append(st+': '+', '.join(m.get('failed_required_gates',[k for k,v in m['gates'].items() if not v])))
            cells.append(dict(architecture='structured_GRU',checkpoint=cp,stratum=st,assay='operational_semantic_full',supports=m['supports'],capture=m['capture'],recovery=m['recovery'],contrast_fraction=m['contrast_fraction'],positive_unit_fraction=m['positive_unit_fraction'],eT_contrast=m['eT_contrast'],eF_contrast=m['eF_contrast'],evidence_id=eid))
            for band,b in m.get('bands',{}).items():bands.append(dict(architecture='structured_GRU',checkpoint=cp,stratum=st,band=band,capture=b['capture'],recovery=b['recovery'],evidence_id=eid,role='DESCRIPTIVE_ACCEPTED_TIME_BAND'))
        rows.append(dict(checkpoint=cp,baseline_eligibility='ELIGIBLE',S1_full_mechanism_support=s['S1']['supports'],S2_full_mechanism_support=s['S2']['supports'],checkpoint_support=d['supports'],end_to_end_support=d['supports'],failure_reason='; '.join(reason) or 'None',canonical_source_audit=eid))
    lr=[]
    for d in lstm['checkpoints']:
        cp=d['checkpoint'];s=d['strata']
        lr.append(dict(checkpoint=cp,baseline_adequacy=baseline[cp],privileged_compact_rank=6,S1_transport_core_support=s['S1']['supports'],S2_transport_core_support=s['S2']['supports'],checkpoint_support=d['supports'],S1_capture=s['S1']['capture'],S1_recovery=s['S1']['recovery'],S2_capture=s['S2']['capture'],S2_recovery=s['S2']['recovery'],operational_semantic_addressability='not established',semantic_specificity='not tested',canonical_source_audit='E13; E19'))
        for st,m in s.items():
            cells.append(dict(architecture='monolithic_LSTM',checkpoint=cp,stratum=st,assay='privileged_transport_core',supports=m['supports'],capture=m['capture'],recovery=m['recovery'],contrast_fraction=m['contrast_fraction'],positive_unit_fraction=m['positive_unit_fraction'],eT_contrast=m['eT_contrast'],eF_contrast=m['eF_contrast'],evidence_id='E19'))
            for band,b in m['bands'].items():bands.append(dict(architecture='monolithic_LSTM',checkpoint=cp,stratum=st,band=band,capture=b['capture'],recovery=b['recovery'],evidence_id='E19',role='DESCRIPTIVE_ACCEPTED_TIME_BAND'))
    counts=dict(original_primary=(sum(x['supports'] for x in original_rows),len(original_rows),'E05'),new_eligible_extension=(sum(x['supports'] for x in ext_rows),len(ext_rows),'E06'),baseline_eligibility=(len(allcp),len(TRAINED),'E06'),end_to_end_support=(sum(r['end_to_end_support'] for r in rows),len(rows),'E06'),conditional_mechanism_support=(sum(d['supports'] for d in allcp.values()),len(allcp),'E06'),lstm_fresh_baseline=(sum(baseline.values()),3,'E13'),lstm_fresh_privileged_transport=(sum(d['supports'] for d in lstm['checkpoints']),3,'E19'))
    nums={k:dict(exact=dict(numerator=a,denominator=b,rate=a/b),rounded_display=f'{a}/{b} ({a/b:.1%})',evidence_ids=[eid]) for k,(a,b,eid) in counts.items()}
    for name,sourcekey in [('end_to_end_support','E10'),('conditional_mechanism_support','M9')]:
        nums[name]['exact']['wilson95']=extension[sourcekey]['wilson95'];nums[name]['wilson95_display']='–'.join(f'{x:.1%}' for x in extension[sourcekey]['wilson95']);nums[name]['interval_role']='Accepted Wilson 95% descriptive interval; checkpoint denominator, no added hypothesis test.'
    nums['constants']={k:dict(exact=v,rounded_display=f'{v:,}',evidence_ids=eids) for k,v,eids in [('GRU_parameters',496200,['E12']),('GRU_carrier_dimension',192,['E01']),('GRU_operational_rank',4,['E01']),('LSTM_parameters',494664,['E12']),('LSTM_carrier_dimension',384,['E12']),('LSTM_privileged_rank',6,['E17','E19'])]}
    nums['GRU_failed_cells']=[dict(checkpoint=c,stratum=st,evidence_id=eid) for c,st,eid in [(291405,'S1','E05'),(291413,'S2','E06')]]
    nums['GRU_291414']=dict(baseline_eligible=False,mechanism_support=None,mechanism_status='NOT_EVALUATED_BASELINE_INELIGIBLE',end_to_end_support=False,evidence_ids=['E06','E08'])
    nums['LSTM_operational_affine_semantic_addressability']=dict(exact='NOT_ROBUSTLY_SUPPORTED',rounded_display='Not robustly supported',evidence_ids=['E16','E17'])
    nums['LSTM_cells']=[dict(checkpoint=d['checkpoint'],stratum=st,exact={k:m[k] for k in ['capture','recovery','contrast_fraction','positive_unit_fraction']},rounded_display={k:f'{m[k]:.6f}' for k in ['capture','recovery','contrast_fraction','positive_unit_fraction']},evidence_ids=['E19']) for d in lstm['checkpoints'] for st,m in d['strata'].items()]
    nums['trained_GRU_checkpoints']=TRAINED;nums['chronology']='Original three locked before seven-model training extension; not ten preregistered from original start.'
    output.mkdir(parents=True,exist_ok=False)
    for name,data,title in [('PAPER2_MODEL_LEVEL_RESULT_TABLE_V1',rows,'Structured-GRU model-level results'),('PAPER2_LSTM_CROSS_ARCHITECTURE_RESULT_TABLE_V1',lr,'LSTM privileged transport results'),('PAPER2_FIGURE_CELL_DATA_V1',cells,'Accepted checkpoint/stratum display data'),('PAPER2_FIGURE_TIME_BAND_DATA_V1',bands,'Accepted descriptive time-band display data')]:
        dump(output/(name+'.csv'),csv_text(data));dump(output/(name+'.md'),md(data,title))
    dump(output/'PAPER2_KEY_NUMBERS_V1.json',nums)
    sourcehashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest().upper() for p in sorted(sources.iterdir()) if p.is_file()}
    dump(output/'TABLE_GENERATION_PROVENANCE.json',dict(source_sha256=sourcehashes,operation='Read accepted booleans/scalars; format rows and arithmetic counts. No model, no gate/statistical recomputation, no new scientific outcome.',files=[p.name for p in sorted(output.iterdir())],generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()))
    print(json.dumps(dict(status='PASS',GRU_rows=len(rows),LSTM_rows=len(lr),cell_rows=len(cells),band_rows=len(bands),counts={k:[a,b] for k,(a,b,eid) in counts.items()})))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();generate(args.sources,args.output)
