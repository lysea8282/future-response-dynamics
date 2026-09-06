"""Read-only byte-integrity verification. Does not import or execute scientific code."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import sys

MAX_BYTES = 50_000_000

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):h.update(block)
    return h.hexdigest().upper()

def validate_entries(root, entries, max_bytes=MAX_BYTES):
    root=Path(root).resolve();seen=set();errors=[]
    for row in entries:
        name=row['path'];rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            errors.append('UNSAFE_PATH:'+name);continue
        if name in seen:errors.append('DUPLICATE_PATH:'+name);continue
        seen.add(name);path=root/name
        if path.is_symlink() or root not in path.resolve().parents:
            errors.append('ESCAPING_PATH:'+name);continue
        if not path.is_file():errors.append('MISSING:'+name);continue
        size=path.stat().st_size
        if size>max_bytes:errors.append('OVERSIZE:'+name)
        if size!=row['size_bytes']:errors.append('SIZE_MISMATCH:'+name)
        if digest(path)!=row['sha256']:errors.append('HASH_MISMATCH:'+name)
    return errors

def sanitize_text(text):
    rules={
        'ABSOLUTE_WINDOWS_PATH':r'(?i)\b[A-Z]:[\\/]',
        'PRIVATE_MOUNT':r'/mnt/' + r'data/',
        'INTERNAL_WORKSPACE_PATH':r'(?i)(?:work_items|release_candidates|Users[\\/]\w+)[\\/]',
        'SECRET_ASSIGNMENT':r'''(?i)(?:api[_-]?key|password|access[_-]?token|client[_-]?secret)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}''',
        'PRIVATE_KEY':r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
        'CREDENTIAL_KEY':r'\bAKIA[A-Z0-9]{16}\b',
        'PERSONAL_ACCESS_TOKEN':r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
        'SERVICE_SECRET':r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b',
    }
    return [name for name,pattern in rules.items() if re.search(pattern,text)]

def inspect(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'provenance/FILE_MANIFEST.json').read_text(encoding='utf-8'))
    errors=validate_entries(root,manifest['files'])
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts and p.name!='FILE_MANIFEST.json'}
    expected={r['path'] for r in manifest['files']}
    if actual!=expected:errors.append('MANIFEST_FILE_SET_MISMATCH')
    for row in manifest['files']:
        p=root/row['path']
        if not p.is_file():continue
        if p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.npz','.pyc'}:errors.append('FORBIDDEN_PAYLOAD:'+row['path'])
        if p.suffix.lower()!='.npy':
            try:text=p.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:errors.append('UNREVIEWED_BINARY:'+row['path']);continue
            errors.extend(label+':'+row['path'] for label in sanitize_text(text))
    return errors

def main():
    root=Path(__file__).resolve().parents[1]
    errors=inspect(root)
    if errors:
        print(json.dumps({'decision':'FAIL','errors':errors},indent=2));return 1
    result=json.loads((root/'provenance/RESULT.json').read_text(encoding='utf-8'))
    print(json.dumps({'decision':'PASS','files':len(json.loads((root/'provenance/FILE_MANIFEST.json').read_text())['files']),'certified_support_counts':result['checkpoint_support_counts'],'scope':'Byte verification and existing result readback only; no metric or model execution'},indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
