"""Portable publication asset verifier. Standard library only; never opens source locators.

Authenticity of the manifest is anchored externally by the review ZIP/Git revision.
This checks integrity and result accounting, not scientific truth or model execution.
"""
from pathlib import Path,PurePosixPath
import argparse,csv,hashlib,json,re,sys,zipfile
from generate_manuscript_tables import build as manuscript_tables
MAX_BYTES=50_000_000
CONTROL={'provenance/ARTIFACT_HASHES.json','provenance/FILE_MANIFEST.json'}
REVIEWED_MEDIA={'figures/section4_1_transport_geometry.pdf':b'%PDF-', 'figures/section4_1_transport_geometry.png':bytes([137,80,78,71,13,10,26,10])}
REVIEWED_MEDIA.update({'figures/paper2_fig2_structured_gru_functional_recovery.pdf':b'%PDF-',
    'figures/paper2_fig2_structured_gru_functional_recovery.png':bytes([137,80,78,71,13,10,26,10])})
REVIEWED_MEDIA.update({f'figures/appendix_b1/panel_{panel}_600dpi.png':b'\x89PNG\r\n\x1a\n' for panel in ('a','b')})
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest().upper()
def validate_entries(root,entries,max_bytes=MAX_BYTES):
    root=Path(root).resolve();errors=[];seen=set()
    for r in entries:
        name=r['path'];part=PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name or not part.parts:errors.append('UNSAFE_PATH:'+name);continue
        if name in seen:errors.append('DUPLICATE_PATH:'+name);continue
        seen.add(name);p=root/name
        if p.is_symlink() or root not in p.resolve().parents:errors.append('UNSAFE_PATH:'+name);continue
        if not p.is_file():errors.append('MISSING:'+name);continue
        size=p.stat().st_size
        if size>max_bytes:errors.append('OVERSIZE:'+name)
        if size!=r['size_bytes']:errors.append('SIZE_MISMATCH:'+name)
        if digest(p)!=r['sha256'].upper():errors.append('HASH_MISMATCH:'+name)
    return errors
def sanitize_text(text):
    rules={'ABSOLUTE_WINDOWS_PATH':r'(?i)\b[A-Z]:[\\/]', 'CREDENTIAL_KEY':r'\bAKIA[A-Z0-9]{16}\b','PERSONAL_ACCESS_TOKEN':r'\bgh[pousr]_[A-Za-z0-9]{30,}\b','PRIVATE_KEY':r'-----BEGIN [A-Z ]*PRIVATE KEY-----','SECRET_ASSIGNMENT':r'''(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}'''}
    return [label for label,pat in rules.items() if re.search(pat,text)]
def publication_files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts and '__pycache__' not in p.relative_to(root).parts}
def inspect(root):
    root=Path(root).resolve();m=json.loads((root/'provenance/ARTIFACT_HASHES.json').read_text(encoding='utf-8'));rows=m['files'];errors=validate_entries(root,rows)
    actual=publication_files(root)-CONTROL;expected={r['path'] for r in rows}
    if actual!=expected:errors.append('MANIFEST_FILE_SET_MISMATCH:'+json.dumps(dict(missing=sorted(expected-actual),extra=sorted(actual-expected))))
    for name in sorted(actual):
        p=root/name
        if p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.npz','.pyc'}:errors.append('FORBIDDEN_PAYLOAD:'+name)
        if p.suffix.lower()=='.npy':continue
        if name=='figures/appendix_b1/lstm_architecture.pptx':
            try:
                with zipfile.ZipFile(p) as archive:
                    for member in archive.namelist():
                        if member.endswith(('.xml','.rels')):
                            errors.extend(tag+':'+name+':'+member for tag in sanitize_text(archive.read(member).decode('utf-8-sig')))
            except (zipfile.BadZipFile,UnicodeError):errors.append('INVALID_REVIEWED_MEDIA:'+name)
            continue
        if name in REVIEWED_MEDIA:
            if not p.read_bytes().startswith(REVIEWED_MEDIA[name]):errors.append('INVALID_REVIEWED_MEDIA:'+name)
            continue
        try:s=p.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:errors.append('UNREVIEWED_BINARY:'+name);continue
        errors.extend(tag+':'+name for tag in sanitize_text(s))
    # Supplemental generated-file provenance bindings are checked locally only.
    manifest=json.loads((root/'provenance/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for r in manifest['files']:
        p=root/r['public_file']
        if not p.is_file() or digest(p)!=r['public_sha256']:errors.append('SOURCE_MANIFEST_PUBLIC_BINDING:'+r['public_file'])
    t=root/'results/paper_tables'
    gr=list(csv.DictReader((t/'PAPER2_MODEL_LEVEL_RESULT_TABLE_V1.csv').open(encoding='utf-8')))
    ids=[int(r['checkpoint']) for r in gr]
    if ids!=[291405,291406,291407,291410,291411,291412,291413,291414,291415,291416]:errors.append('GRU_TRAINED_DENOMINATOR')
    by={int(r['checkpoint']):r for r in gr}
    if len(by)!=10:errors.append('GRU_DUPLICATE_CHECKPOINT')
    if 291414 in by:
        r=by[291414]
        if (r['baseline_eligibility'],r['S1_full_mechanism_support'],r['S2_full_mechanism_support'],r['checkpoint_support'],r['end_to_end_support'])!=('INELIGIBLE','NOT_EVALUATED','NOT_EVALUATED','NOT_EVALUATED','False'):errors.append('BASELINE_INELIGIBLE_CONFLATED')
    if sum(r['end_to_end_support']=='True' for r in gr)!=7 or sum(r['baseline_eligibility']=='ELIGIBLE' for r in gr)!=9:errors.append('GRU_SUPPORT_COUNTS')
    for cp,st in [(291405,'S1'),(291413,'S2')]:
        if cp not in by or by[cp][st+'_full_mechanism_support']!='False':errors.append('GRU_FAILURE_REMOVED')
    ls=list(csv.DictReader((t/'PAPER2_LSTM_CROSS_ARCHITECTURE_RESULT_TABLE_V1.csv').open(encoding='utf-8')))
    if [int(r['checkpoint']) for r in ls]!=[392001,392002,392003]:errors.append('LSTM_CHECKPOINT_SET')
    if any(r['operational_semantic_addressability']!='not established' or r['semantic_specificity']!='not tested' or r['checkpoint_support']!='True' or r['baseline_adequacy']!='True' for r in ls):errors.append('LSTM_SCOPE_OR_SUPPORT')
    tables,numerical=manuscript_tables(root)
    for name,table in tables.items():
        with (root/'results/manuscript_tables'/(name+'.csv')).open(encoding='utf-8',newline='') as stream:
            if list(csv.DictReader(stream))!=[{k:str(v) for k,v in row.items()} for row in table]:errors.append('MANUSCRIPT_TABLE_READBACK:'+name)
    if json.loads((root/'results/manuscript_tables/path_accounting.json').read_text())!=numerical:errors.append('PATH_ACCOUNTING_READBACK')
    return errors
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args()
    try:errors=inspect(a.root)
    except (OSError,ValueError,KeyError,TypeError) as exc:errors=['UNREADABLE_OR_INVALID_PACKAGE:'+str(exc)]
    print(json.dumps(dict(decision='FAIL' if errors else 'PASS',errors=errors,scope='Publication bytes, local provenance bindings, denominator accounting. No private-path access; no neural execution.'),indent=2));return int(bool(errors))
if __name__=='__main__':sys.exit(main())
