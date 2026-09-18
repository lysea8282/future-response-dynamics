"""Executed positive/negative packaging tests; no scientific computation."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import tempfile
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

spec=importlib.util.spec_from_file_location('verify_artifacts',Path(__file__).resolve().parents[1]/'scripts/verify_artifacts.py')
verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
SCRATCH=None

class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='integrity_',dir=SCRATCH)
        self.root=Path(self.tmp.name)
        (self.root/'small.txt').write_bytes(b'fixture\n')
        self.row={'path':'small.txt','size_bytes':8,'sha256':hashlib.sha256(b'fixture\n').hexdigest().upper()}
    def tearDown(self):self.tmp.cleanup()
    def test_exact_fixture_passes(self):self.assertEqual(verify.validate_entries(self.root,[self.row]),[])
    def test_modified_bytes_fail(self):
        (self.root/'small.txt').write_bytes(b'changed\n')
        self.assertTrue(any(x.startswith('HASH_MISMATCH:') for x in verify.validate_entries(self.root,[self.row])))
    def test_missing_file_fails(self):
        self.assertTrue(any(x.startswith('MISSING:') for x in verify.validate_entries(self.root,[{**self.row,'path':'missing.txt'}])))
    def test_parent_escape_fails(self):
        self.assertTrue(any(x.startswith('UNSAFE_PATH:') for x in verify.validate_entries(self.root,[{**self.row,'path':'../escape.txt'}])))
    def test_absolute_path_fails(self):
        self.assertTrue(any(x.startswith('UNSAFE_PATH:') for x in verify.validate_entries(self.root,[{**self.row,'path':'/escape.txt'}])))
    def test_duplicate_entry_fails(self):
        self.assertTrue(any(x.startswith('DUPLICATE_PATH:') for x in verify.validate_entries(self.root,[self.row,self.row])))
    def test_size_limit_fails(self):
        self.assertTrue(any(x.startswith('OVERSIZE:') for x in verify.validate_entries(self.root,[self.row],max_bytes=7)))
    def test_metadata_size_mismatch_fails(self):
        self.assertTrue(any(x.startswith('SIZE_MISMATCH:') for x in verify.validate_entries(self.root,[{**self.row,'size_bytes':9}])))
    def test_private_path_scan(self):
        self.assertIn('ABSOLUTE_WINDOWS_PATH',verify.sanitize_text('D'+':'+chr(92)+'private'+chr(92)+'file'))
    def test_secret_scan(self):
        self.assertIn('CREDENTIAL_KEY',verify.sanitize_text('AK'+'IA'+'X'*16))
    def test_ordinary_scientific_identifier_allowed(self):
        self.assertEqual(verify.sanitize_text('checkpoint_291404 Delta_PF_Align POST_HOC_DESCRIPTIVE'),[])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args();args.scratch.mkdir(parents=True,exist_ok=True);SCRATCH=args.scratch
    unittest.main(argv=['test_integrity'],verbosity=2)
